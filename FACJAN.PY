import os, json

data_root = "data"
grid_root = "data/grid_data"

train_json = "data/splits/mug_augmented.json"
val_json   = "data/splits/mug_1.json"

def peek(js):
    d = json.load(open(js))
    # estrai lista dal livello dove il loader guarda (di solito d["acronym"])
    lst = d.get("acronym", d)
    # se è ancora dict, forse è {"acronym":{"mug":[...]}}
    if isinstance(lst, dict):
        lst = lst.get("mug", [])
    return lst

train_list = peek(train_json)
val_list   = peek(val_json)

print("[TRAIN] count:", len(train_list), "sample:", train_list[:2])
print("[VALID] count:", len(val_list),   "sample:", val_list[:2])

def exists_pair(rel):
    gt   = os.path.join(data_root, rel, "sdf_data.csv")
    grid = os.path.join(grid_root, rel, "grid_gt.csv")
    return os.path.exists(gt), os.path.exists(grid), gt, grid

# prova 3 esempi per ciascuno
for tag,lst in [("TRAIN",train_list),("VALID",val_list)]:
    print(f"\n[{tag}] existence check:")
    for rel in lst[:3]:
        ok_gt, ok_grid, gt_p, grid_p = exists_pair(rel)
        print(f"- {rel}\n   GT  {ok_gt} → {gt_p}\n   GRID {ok_grid} → {grid_p}")
