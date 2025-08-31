import os, sys, time, shutil, hashlib, json, csv, resource, signal, contextlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from bps import bps
from . import base  # la tua base.Dataset

# ---------- timeout helper (Linux / main thread) ----------
@contextlib.contextmanager
def time_limit(seconds: int, on_file: str = ""):
    def handler(signum, frame):
        raise TimeoutError(f"Timeout ({seconds}s) while reading: {on_file}")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ---------- diagnostica ----------
import faulthandler
faulthandler.enable()

def _ram_gb():
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return rss_kb / 1024 / 1024
        else:
            return rss_kb / 1024 / 1024
    except Exception:
        return -1.0

def _gpu_mem_gb():
    try:
        if torch.cuda.is_available():
            return (torch.cuda.memory_allocated()/1e9,
                    torch.cuda.memory_reserved()/1e9,
                    torch.cuda.max_memory_allocated()/1e9)
    except Exception:
        pass
    return (0.0, 0.0, 0.0)

def _open_fds():
    try:
        return len(os.listdir('/proc/self/fd'))
    except Exception:
        return -1

def _disk_free_gb(path="/"):
    try:
        total, used, free = shutil.disk_usage(path)
        return free / 1e9
    except Exception:
        return -1.0

def log_status(tag):
    ga, gr, gmax = _gpu_mem_gb()
    print(f"[{time.strftime('%H:%M:%S')}] {tag} | "
          f"RAM={_ram_gb():.2f}GB | GPU(a/r/max)={ga:.2f}/{gr:.2f}/{gmax:.2f}GB | "
          f"fds={_open_fds()} | disk_free={_disk_free_gb('/'):.1f}GB",
          flush=True)


class SdfLoader(base.Dataset):
    def __init__(self,
                 data_source,
                 split_file,
                 grid_source=None,
                 samples_per_mesh=16000,
                 pc_size=1024,
                 modulation_path=None,
                 cache_dir=".sdf_cache",
                 csv_timeout_s=60,
                 skip_bad_files=True,
                 verbose_every=1):

        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.grid_source = grid_source
        self.cache_dir = cache_dir
        self.csv_timeout_s = int(csv_timeout_s)
        self.skip_bad_files = bool(skip_bad_files)
        self.verbose_every = max(1, int(verbose_every))
        os.makedirs(self.cache_dir, exist_ok=True)

        # ----- paths -----
        self.gt_files = self.get_instance_filenames(
            data_source, split_file, filter_modulation_path=modulation_path
        )
        self.paths = list(self.gt_files)

        if grid_source:
            self.grid_files = self.get_instance_filenames(
                grid_source, split_file, gt_filename="grid_gt.csv",
                filter_modulation_path=modulation_path
            )
            assert len(self.grid_files) == len(self.gt_files), "grid/gt size mismatch"
        else:
            self.grid_files = None

        print(f"[SdfLoader] inizializzazione dataset con {len(self.gt_files)} esempi. "
              f"Grid={bool(self.grid_source)}", flush=True)
        log_status("init.start")

        # ----- BPS grid -----
        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5).contiguous()
        self._bps_grid_np = self.bps_grid.cpu().numpy()

        # ----- arrays finali -----
        kept_gt_tensors, kept_bps, kept_paths = [], [], []
        kept_grid_tensors = [] if self.grid_source else None

        n_total = len(self.gt_files)
        print(f"[SdfLoader] caching {n_total} items…", flush=True)

        # faulthandler trace periodico per eventuali hang
        faulthandler.dump_traceback_later(120, repeat=True)

        slow_read_warn_s = 2.0
        slow_bps_warn_s  = 3.0
        last_status_t    = time.time()

        for i, f in enumerate(self.gt_files):
            try:
                fsize_mb = os.path.getsize(f)/1e6 if os.path.exists(f) else -1
            except Exception:
                fsize_mb = -1

            if (i % self.verbose_every) == 0:
                print(f"[{i+1}/{n_total}] GT: {f} ({fsize_mb:.2f} MB)", flush=True)

            # 1) GT csv -> tensor (cache + timeout + fallback)
            t0 = time.time()
            gt_tensor = self._load_or_cache_csv(f, key="gt")
            t1 = time.time()
            if gt_tensor is None:
                msg = f"[warn] skip GT (None/failed): {f}"
                print(msg, flush=True)
                if not self.skip_bad_files:
                    raise RuntimeError(msg)
                continue
            if (t1 - t0) > slow_read_warn_s:
                print(f"[warn][CSV slow {t1-t0:.2f}s] {f}", flush=True)

            # 2) pointcloud -> BPS (cache)
            t2 = time.time()
            try:
                pc = self.get_pointcloud(gt_tensor, load_from_path=False)
                bps_tensor = self._load_or_cache_bps(f, pc)
                if bps_tensor is None:
                    raise RuntimeError("BPS is None")
            except Exception as e:
                msg = f"[warn] skip BPS fail for {f}: {e}"
                print(msg, flush=True)
                if not self.skip_bad_files:
                    raise
                continue
            t3 = time.time()
            if (t3 - t2) > slow_bps_warn_s:
                print(f"[warn][BPS slow {t3-t2:.2f}s] {f}", flush=True)

            # 3) grid csv (se richiesto)
            if self.grid_source:
                gf = self.grid_files[i]
                try:
                    gfsize_mb = os.path.getsize(gf)/1e6 if os.path.exists(gf) else -1
                except Exception:
                    gfsize_mb = -1
                if (i % self.verbose_every) == 0:
                    print(f"          GRID: {gf} ({gfsize_mb:.2f} MB)", flush=True)

                tg0 = time.time()
                grid_tensor = self._load_or_cache_csv(gf, key="grid")
                tg1 = time.time()
                if grid_tensor is None:
                    msg = f"[warn] skip pair (grid failed): {gf}"
                    print(msg, flush=True)
                    if not self.skip_bad_files:
                        raise RuntimeError(msg)
                    continue
                if (tg1 - tg0) > slow_read_warn_s:
                    print(f"[warn][GRID CSV slow {tg1-tg0:.2f}s] {gf}", flush=True)

            # tieni ciò che è ok
            kept_gt_tensors.append(gt_tensor)
            kept_bps.append(bps_tensor)
            kept_paths.append(f)
            if self.grid_source:
                kept_grid_tensors.append(grid_tensor)

            # log periodico di stato
            if time.time() - last_status_t > 60:
                log_status(f"init.progress i={i+1}/{n_total}")
                last_status_t = time.time()

        # chiudi trace periodico
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

        self.gt_tensors = kept_gt_tensors
        self.preprocessed_bps = kept_bps
        self.paths = kept_paths
        self.grid_tensors = kept_grid_tensors if self.grid_source else None

        print(f"[SdfLoader] kept {len(self.gt_tensors)} / {n_total} items.", flush=True)
        log_status("init.done")

    # ---------------- cache helpers ----------------
    def _hash_path(self, path):
        try:
            st = os.stat(path)
            meta = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            meta = path
        return hashlib.sha1(meta.encode("utf-8")).hexdigest()

    def _cache_file(self, path, kind):
        h = self._hash_path(path)
        return os.path.join(self.cache_dir, f"{kind}_{h}.pt")

    def _fast_read_csv(self, csv_path: str):
        # 1) tenta engine='c'
        try:
            with time_limit(self.csv_timeout_s, on_file=csv_path):
                arr = pd.read_csv(csv_path, sep=",", header=None,
                                  dtype=np.float32, engine="c").values
        except TimeoutError as e:
            print(f"[timeout] {e}", flush=True)
            return None
        except Exception as e_c:
            print(f"[read_csv warn] '{csv_path}' engine='c' failed: {e_c} -> fallback python", flush=True)
            try:
                with time_limit(self.csv_timeout_s, on_file=csv_path):
                    arr = pd.read_csv(csv_path, sep=",", header=None,
                                      dtype=np.float32, engine="python").values
            except TimeoutError as e2:
                print(f"[timeout] {e2}", flush=True)
                return None
            except Exception as e_py:
                print(f"[read_csv fail] {csv_path}: {e_py}", flush=True)
                return None

        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            print(f"[invalid CSV shape] {csv_path}: {arr.shape}", flush=True)
            return None

        return torch.from_numpy(np.ascontiguousarray(arr))

    def _load_or_cache_csv(self, csv_path, key="gt"):
        cache_path = self._cache_file(csv_path, key)
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception as e:
                print(f"[cache load fail] {cache_path}: {e}; re-reading CSV.", flush=True)

        tens = self._fast_read_csv(csv_path)
        if tens is not None:
            try:
                torch.save(tens, cache_path)
            except Exception as e:
                print(f"[cache save fail] {cache_path}: {e}", flush=True)
        return tens

    def _load_or_cache_bps(self, src_path, pointcloud_tensor):
        cache_path = self._cache_file(src_path, "bps")
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception as e:
                print(f"[cache load fail] {cache_path}: {e}; recomputing BPS.", flush=True)

        try:
            bps_tensor = self._get_base_points_from_pc(pointcloud_tensor)  # (3,32,32,32)
            torch.save(bps_tensor, cache_path)
            return bps_tensor
        except Exception as e:
            print(f"[bps fail] {src_path}: {e}", flush=True)
            return None

    # ---------------- dataset API ----------------
    def __len__(self):
        return len(self.gt_tensors)

    def __getitem__(self, idx):
        t0 = time.time()
        near_n = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            self.gt_tensors[idx], near_n, self.pc_size, load_from_path=False
        )
        t1 = time.time()

        basis_point = self.preprocessed_bps[idx]
        grid = None
        if self.grid_source:
            grid_count = self.samples_per_mesh - near_n
            _, grid_xyz, grid_gt = self.labeled_sampling(
                self.grid_tensors[idx], grid_count, pc_size=grid_count, load_from_path=False
            )
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz), dim=0)
            sdf_gt  = torch.cat((sdf_gt,  grid_gt),  dim=0)
            grid = self.get_grid(self.grid_tensors[idx], load_from_path=False)

        t2 = time.time()
        # log ogni ~1% degli item
        if (idx % max(1, int(len(self) * 0.01))) == 0:
            print(f"[getitem] idx={idx} sample={t1-t0:.3f}s merge={t2-t1:.3f}s total={t2-t0:.3f}s", flush=True)
            log_status(f"getitem.idx={idx}")

        return {
            "xyz":         sdf_xyz.float().contiguous().squeeze(),
            "gt_sdf":      sdf_gt.float().contiguous().squeeze(),
            "basis_point": basis_point.float().contiguous().squeeze(),
            "grid_point":  (grid.float().contiguous().squeeze()
                            if (self.grid_source and grid is not None) else None),
            "point_cloud": pc.float().contiguous().squeeze(),
            "paths":       self.paths[idx],
        }

    # ---------------- utilities ----------------
    def _create_bps_grid(self, grid_size=32, radius=1.5):
        bps_grid_np = bps.generate_grid_basis(
            grid_size=grid_size, n_dims=3, minv=-radius, maxv=radius
        ).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(bps_grid_np))

    def _get_base_points_from_pc(self, pointcloud: torch.Tensor) -> torch.Tensor:
        pc_np = pointcloud.detach().cpu().numpy().astype(np.float32, copy=False)
        pc_np = pc_np[np.newaxis, ...]  # (1,N,3)
        pc_norm = bps.normalize(pc_np)
        x_bps = bps.encode(
            pc_norm,
            bps_arrangement="custom",
            custom_basis=self._bps_grid_np,
            bps_cell_type="deltas",
            n_jobs=1
        )  # (1, 32*32*32, 3)
        x_bps = x_bps.reshape(1, 32, 32, 32, 3).transpose(0, 4, 1, 2, 3)  # (1,3,32,32,32)
        return torch.from_numpy(np.ascontiguousarray(x_bps)).float().squeeze(0)
