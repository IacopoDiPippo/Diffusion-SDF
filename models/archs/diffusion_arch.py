import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from rotary_embedding_torch import RotaryEmbedding

from diff_utils.model_utils import * 

from random import sample

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth,
        dim_in_out=None,
        cross_attn=False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True, 
        normformer = False,
        rotary_emb = True, 
        **kwargs
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        point_feature_dim = kwargs.get('point_feature_dim', dim)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
        else:
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, out_dim=dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn

    def forward(self, x, time_emb=None, context=None):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            #assert context is not None 
            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now 
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x 
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now 
                #print("x2 shape, context shape: ", x.shape, context.shape)
                
                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x
        
        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias)
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x)
        return self.project_out(out)

class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim,
        dim_in_out=None,
        num_timesteps = None,
        num_time_embeds = 1,
        cond = None,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = kwargs.get('cross_attn', False)
        self.cond_dropout = kwargs.get('cond_dropout', False)
        self.point_feature_dim = kwargs.get('point_feature_dim', dim)

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim = dim, dim_in_out=self.dim_in_out, **kwargs)
        # adattatore opzionale per cond globali (B, D_latent) -> (B, 1, point_feature_dim)
        self.cond_proj = None
        if cond:
            if self.cond_dropout:
                self.pointnet = ConvPointnet(c_dim=self.point_feature_dim)
            # output dim of pointnet needs to match model dim; unless add additional linear layer
            else:
                self.pointnet = PointEncoder(in_channels=3, latent_dim=self.point_feature_dim) 


    def forward(
        self,
        data, 
        diffusion_timesteps,
        pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        if self.cond:
            assert isinstance(data, tuple), "Con cond=True, 'data' deve essere (x_t, cond)"
            data, cond = data  # data: [B, D]; cond: può essere [B, N, 3] (PC) OPPURE [B, D_latent] (latente globale)

            # 1) ricava un Tensor 'cond_feature' su device con shape o (B, D_*) o (B, N, D_*)
            if isinstance(cond, (list, tuple)):
                cond = cond[0]
            if not torch.is_tensor(cond):
                cond = torch.as_tensor(cond)

            cond = cond.to(data.device).float()

            # a) caso POINT CLOUD: (B, N, 3) --> estrai embedding con PointNet/PointEncoder (tipicamente (B, D_ctx))
            if cond.ndim == 3 and cond.shape[-1] == 3:
                # se usi PointEncoder.encode restituisce (B, D) oppure (B, N_ctx, D) in base alla tua implementazione
                if self.cond_dropout:
                    # classifier-free guidance (se attivo): qui potresti mettere la tua logica prob/percentage
                    cond_feature = self.pointnet(cond, cond)  # tendente a (B, N_ctx, D_ctx)
                else:
                    cond_feature = self.pointnet.encode(cond)  # spesso (B, D_ctx)

            # b) caso LATENTE GLOBALE: (B, D_latent) --> usalo direttamente
            elif cond.ndim == 2:
                cond_feature = cond  # (B, D_latent)

            # c) caso già tokenizzato: (B, N_ctx, D_ctx)
            elif cond.ndim == 3:
                cond_feature = cond

            else:
                raise ValueError(f"Shape cond inattesa: {tuple(cond.shape)}")

            # 2) porta SEMPRE a forma (B, N_ctx, D_ctx): se è (B, D) diventa (B, 1, D)
            if cond_feature.dim() == 2:
                cond_feature = cond_feature.unsqueeze(1)  # (B, 1, D_ctx)

            # 3) adatta la dim di feature a quella attesa dal cross-attn (point_feature_dim)
            D_in = cond_feature.size(-1)
            if D_in != self.point_feature_dim:
                if (self.cond_proj is None) or (self.cond_proj.in_features != D_in) or (self.cond_proj.out_features != self.point_feature_dim):
                    self.cond_proj = nn.Linear(D_in, self.point_feature_dim, bias=False).to(cond_feature.device)
                cond_feature = self.cond_proj(cond_feature)

            # 4) (opzionale) classifier-free guidance: azzera per ramo unconditional
            if self.cond_dropout:
                # Esempio semplice: 20% unconditional
                prob = torch.randint(0, 10, (1,), device=data.device)
                if prob < 2 or pass_cond == 0:
                    cond_feature = torch.zeros_like(cond_feature)
                elif pass_cond == 1:
                    pass  # lascia cond_feature intatto

        else:
            cond_feature = None


        print("Cond_feature shape", cond_feature.shape)
        batch, dim, device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)

        data = data.unsqueeze(1)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]

        if self.cond and not self.cross_attn:
            model_inputs.insert(0, cond_feature) # cond_feature defined in first loop above 
        
        tokens = torch.cat(model_inputs, dim = 1) # (b, 3/4, d); batch and d=512 same across the model_inputs 
        #print("tokens shape: ", tokens.shape)

        if self.cross_attn:
            cond_feature = None if not self.cond else cond_feature
            #print("tokens shape: ", tokens.shape, cond_feature.shape)
            tokens = self.causal_transformer(tokens, context=cond_feature)
        else:
            tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :]

        return pred

