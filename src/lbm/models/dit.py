"""LBM policy: AdaLN-Zero DiT with vision cross-attention."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.config import DiTConfig
from lbm.dataloader.pad import DEFAULT_EMBODIMENT_ID
from lbm.models.attention import attn_out, configure_torch_sdp, split_kv, split_q, split_qkv
from lbm.models.common import DiTMlp, FeedForward, run_maybe_frozen
from lbm.models.encoders import build_language_encoder, build_vision_backbone
from lbm.models.weights import load_into


def modulate(x, shift, scale):
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate, residual):
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def conditioning_chunks(layer, conditioning, count):
    """Project repeated prefix/non-prefix conditions once per sample."""
    if isinstance(conditioning, tuple):
        normal, prefix, mask = conditioning
        normal = layer(normal).chunk(count, dim=-1)
        prefix = layer(prefix).chunk(count, dim=-1)
        return tuple(torch.where(mask[..., None], p[:, None], n[:, None])
                     for n, p in zip(normal, prefix))
    return layer(conditioning).chunk(count, dim=-1)


def get_1d_sincos_pos_embed(embed_dim, length):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t):
        freqs = self.freqs
        if freqs.device != t.device:
            freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_shape = t.shape
        t_freq = self.timestep_embedding(t.reshape(-1))
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb.reshape(*t_shape, -1)


class DiTAttention(nn.Module):
    """Self-attention over action tokens."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        q, k, v = split_qkv(self.qkv(x), self.num_heads, self.head_dim)
        return self.proj(attn_out(q, k, v, self.num_heads, self.head_dim))


class DiTCrossAttention(nn.Module):
    """Vision cross-attention."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, context):
        q = split_q(self.q(x), self.num_heads, self.head_dim)
        k, v = split_kv(self.kv(context), self.num_heads, self.head_dim)
        return self.proj(attn_out(q, k, v, self.num_heads, self.head_dim))


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block with vision cross-attention (9-way modulation)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        ffn_hidden = int(hidden_size * mlp_ratio)
        self.mlp = DiTMlp(hidden_size, ffn_hidden)
        self.norm_xattn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_xattn_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = DiTCrossAttention(hidden_size, num_heads)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, vision_tokens):
        (
            shift_msa, scale_msa, gate_msa,
            shift_xattn, scale_xattn, gate_xattn,
            shift_mlp, scale_mlp, gate_mlp,
        ) = conditioning_chunks(self.adaLN_modulation, c, 9)

        x = x + gate_residual(gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))

        x_normed = modulate(self.norm_xattn(x), shift_xattn, scale_xattn)
        kv = self.norm_xattn_kv(vision_tokens)
        x = x + gate_residual(gate_xattn, self.cross_attn(x_normed, kv))

        x = x + gate_residual(gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class DiTBlockGroup(nn.Module):
    """Consecutive ``DiTBlock``s treated as one Megatron-FSDP unit.

    Larger units amortize NCCL launch latency so parameter all-gather can hide
    behind more compute. All groups should have the same block count so FSDP
    double-buffering stays size-symmetric.
    """

    def __init__(self, blocks: list[DiTBlock]):
        super().__init__()
        if not blocks:
            raise ValueError("DiTBlockGroup requires at least one DiTBlock")
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, c, vision_tokens):
        for block in self.blocks:
            x = block(x, c, vision_tokens)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = conditioning_chunks(self.adaLN_modulation, c, 2)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class AttentionPoolBlock(nn.Module):
    """Learnable queries cross-attend to ViT tokens (per camera)."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(mlp_ratio * embed_dim))

    def forward(self, x, queries):
        q = split_q(self.q(self.ln_1(queries)), self.num_heads, self.head_dim)
        k, v = split_kv(self.kv(self.ln_1(x)), self.num_heads, self.head_dim)
        out = self.out_proj(attn_out(q, k, v, self.num_heads, self.head_dim))
        return self.mlp(self.ln_2(out)) + out


class VisionTokenPool(nn.Module):
    """Per-camera attention pool + proj.

    Megatron-FSDP only all-gathers parameters on ``forward()``. Query tensors
    used to live in a ``ParameterDict`` on ``DiTPolicy`` and were indexed in
    Python, so they stayed sharded (storage size 0) once frozen DINO no longer
    shared a leftover bucket with them. This module is an FSDP unit: its
    pre-forward hook unshards queries, pool, and proj together.
    """

    def __init__(self, config: DiTConfig, camera_keys: list[str]):
        super().__init__()
        self.camera_keys = list(camera_keys)
        H = config.hidden_size
        self.queries = nn.ParameterDict(
            {
                cam: nn.Parameter(
                    torch.randn(1, config.vision_pool_num_queries, config.vit_embed_dim) * 0.02
                )
                for cam in self.camera_keys
            }
        )
        self.apool = nn.ModuleDict(
            {
                cam: AttentionPoolBlock(
                    config.vit_embed_dim,
                    config.vision_pool_num_heads,
                    config.vision_pool_mlp_ratio,
                )
                for cam in self.camera_keys
            }
        )
        self.proj = nn.Linear(config.vit_embed_dim, H)
        self.camera_embed = nn.Embedding(len(self.camera_keys), H)

    def forward(self, tokens_by_cam: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = []
        for cam in self.camera_keys:
            tokens = tokens_by_cam[cam].to(self.queries[cam].dtype)
            queries = self.queries[cam].expand(tokens.shape[0], -1, -1)
            pooled.append(self.apool[cam](tokens, queries))
        tokens_by_camera = torch.stack(pooled, dim=1)
        B, Nc, K, D = tokens_by_camera.shape
        vision_tokens = self.proj(
            tokens_by_camera.reshape(B * Nc * K, D).to(self.proj.weight.dtype)
        ).reshape(B, Nc, K, -1)
        cam_emb = self.camera_embed(torch.arange(Nc, device=vision_tokens.device))
        return (vision_tokens + cam_emb[None, :, None, :]).reshape(B, Nc * K, -1)


class LanguagePoolProj(nn.Module):
    """Pool language encoder outputs and project into DiT hidden.

    Original CLIP / precomputed ``task_vec_clip`` are already ``(B, D)``.
    Original T5 encoder is ``(B, T, D)`` — mean-pool with the pad mask, then
    the same linear as vision's ``vision_tokens_proj``.
    """

    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size)

    def forward(self, x, attention_mask=None):
        if x.ndim == 3:
            if attention_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = attention_mask.to(dtype=x.dtype)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = x
        return self.proj(pooled.to(self.proj.weight.dtype))


class DiTPolicy(nn.Module):
    """LBM: flow-matching DiT policy with vision cross-attention."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        configure_torch_sdp(mode="auto")
        self.config = config
        H = config.hidden_size
        self.camera_keys = list(config.camera_keys)
        self.chunk_length = config.chunk_length
        self.action_dim = config.action_dim

        self.x_embedder = nn.Linear(config.state_dim, H)
        self.y_embedder = nn.Linear(config.action_dim, H)
        self.embodiment_embed = nn.Embedding(DEFAULT_EMBODIMENT_ID + 1, H)
        nn.init.zeros_(self.embodiment_embed.weight)
        # Checkpoint compatibility only; unused in forward.
        self.img_proj = nn.Linear(config.vit_embed_dim, H)
        self.img_proj.requires_grad_(False)
        self.t_embedder = TimestepEmbedder(H)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.chunk_length, H), requires_grad=False)

        self.img_backbone = build_vision_backbone(config)
        self.language_encoder = build_language_encoder(config)
        task_dim = (
            self.language_encoder.out_dim
            if self.language_encoder is not None
            else config.task_embed_dim
        )

        self.vision_pool = VisionTokenPool(config, self.camera_keys)
        self.language_pool_proj = LanguagePoolProj(task_dim, H)
        self.blocks = nn.ModuleList(
            DiTBlock(H, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.final_layer = FinalLayer(H, config.action_dim)

        # cond = [state, task, timestep] -> hidden (vision goes via cross-attn)
        self.cond_proj = nn.Sequential(
            nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H), nn.LayerNorm(H)
        )

        self.register_buffer("clip_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        pos = get_1d_sincos_pos_embed(H, config.chunk_length)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self._apply_encoder_trainable()

    @property
    def apool(self):
        return self.vision_pool.apool

    @property
    def apool_queries(self):
        return self.vision_pool.queries

    @property
    def vision_tokens_proj(self):
        return self.vision_pool.proj

    @property
    def vision_camera_embed(self):
        return self.vision_pool.camera_embed

    def _apply_encoder_trainable(self) -> None:
        cfg = self.config
        self.img_backbone.requires_grad_(cfg.train_vision_encoder)
        if not cfg.train_vision_encoder:
            self.img_backbone.eval()
        if self.language_encoder is not None:
            self.language_encoder.requires_grad_(cfg.train_language_encoder)
            if not cfg.train_language_encoder:
                self.language_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.config.train_vision_encoder:
            self.img_backbone.eval()
        if self.language_encoder is not None and not self.config.train_language_encoder:
            self.language_encoder.eval()
        return self

    def encode_task(self, batch) -> torch.Tensor:
        """Language features after pool+proj: ``(B, hidden)``.

        ``language_encoder=none`` uses ``task_vec_clip``. CLIP returns a vector
        (original encode_text). T5 returns token states (original encoder).
        """
        mask = batch.get("task_token_mask")
        if self.language_encoder is None:
            x = batch["task_vec_clip"]
        else:
            x = run_maybe_frozen(
                self.config.train_language_encoder,
                lambda: self.language_encoder(batch["task_tokens"], attention_mask=mask),
            )
        return self.language_pool_proj(x, attention_mask=mask)

    def encode_vision_features(self, images):
        """Raw backbone tokens (B, history, patches, width), before trainable pooling."""
        first = images[self.camera_keys[0]]
        bsz = first.shape[0]
        history = first.shape[1] if first.ndim == 5 else 1
        features = {}
        for cam in self.camera_keys:
            image = images[cam]
            flat = image.reshape(bsz * history, *image.shape[-3:])
            tokens = run_maybe_frozen(
                self.config.train_vision_encoder,
                lambda flat=flat: self.img_backbone.encode_image_tokens(flat),
            )
            features[cam] = tokens.reshape(bsz, history, *tokens.shape[1:])
        return features

    def build_vision_tokens(self, images=None, camera_mask=None, *, features=None):
        """Pool live images or frozen-backbone cache tokens, then apply camera masks."""
        if features is not None:
            if self.config.train_vision_encoder:
                raise ValueError("cached vision features require a frozen vision encoder")
            if images is not None:
                raise ValueError("provide images or vision_features, not both")
        else:
            features = self.encode_vision_features(images)
        if set(features) != set(self.camera_keys):
            raise ValueError("vision feature cameras must match the model")
        first = features[self.camera_keys[0]]
        if first.ndim != 4 or first.shape[-1] != self.config.vit_embed_dim:
            raise ValueError("vision features must have shape (B, history, patches, vit_embed_dim)")
        bsz, t_hist = first.shape[:2]
        time_expand = t_hist > 1
        tokens_by_cam = {}
        for cam in self.camera_keys:
            feature = features[cam]
            if feature.shape != first.shape:
                raise ValueError("vision feature shapes must agree across cameras")
            tokens_by_cam[cam] = feature.reshape(bsz * t_hist, *feature.shape[2:])
        tokens = self.vision_pool(tokens_by_cam)
        nq = self.config.vision_pool_num_queries
        nc = len(self.camera_keys)
        if time_expand:
            tokens = tokens.reshape(bsz, t_hist, nc, nq, tokens.shape[-1])
            if camera_mask is not None:
                tokens = tokens * camera_mask.to(tokens.dtype)[:, None, :, None, None]
            tokens = tokens.reshape(bsz, t_hist * nc * nq, tokens.shape[-1])
        elif camera_mask is not None:
            hidden = tokens.shape[-1]
            tokens = tokens.reshape(tokens.shape[0], nc, nq, hidden)
            tokens = tokens * camera_mask.to(tokens.dtype)[:, :, None, None]
            tokens = tokens.reshape(tokens.shape[0], nc * nq, hidden)
        return tokens

    def compute_cond(self, state, task_h, t_cond, embodiment_id=None):
        """state (B, D); task_h (B, hidden) from ``encode_task``; t_cond (B,) or (B,T).
        Returns conditioning c: (B,H) or (B,T,H)."""
        model_dtype = self.x_embedder.weight.dtype
        cond_dtype = self.cond_proj[0].weight.dtype
        st_vec = self.x_embedder(state.to(model_dtype))
        if embodiment_id is None:
            embodiment_id = torch.full(
                (state.shape[0],),
                DEFAULT_EMBODIMENT_ID,
                device=state.device,
                dtype=torch.long,
            )
        st_vec = st_vec + self.embodiment_embed(embodiment_id.long()).to(st_vec.dtype)
        task_vec_h = task_h.to(model_dtype)
        t_vec = self.t_embedder(t_cond.to(model_dtype))
        cond_parts = [st_vec, task_vec_h, t_vec]
        if t_vec.ndim == 3:
            T = t_vec.shape[1]
            cond_parts = [
                p.unsqueeze(1).expand(-1, T, -1) if p.ndim == 2 else p for p in cond_parts
            ]
        cond_concat = torch.cat(cond_parts, dim=-1).to(cond_dtype)
        if cond_dtype == torch.float32 and cond_concat.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.cond_proj(cond_concat).to(model_dtype)
        return self.cond_proj(cond_concat).to(model_dtype)

    def predict_velocity(self, x_t, c, vision_tokens):
        z = self.y_embedder(x_t) + self.pos_embed.data[:, : x_t.shape[1], :]
        for block in self.blocks:
            z = block(z, c, vision_tokens)
        return self.final_layer(z, c)

    def forward(
        self,
        batch,
        noise=None,
        t=None,
        max_action_prefix=0,
        prefix_conditioning_prob=1.0,
        prefix_noise_scale=0.0,
        compact_prefix_conditioning=True,
        sample_steps=None,
    ):
        """Flow-matching training loss with optional action-prefix conditioning.
        batch: state (B,14), images dict, actions (B,50,14), task_vec_clip (B,512),
        optional state_is_masked (B,) bool."""
        if sample_steps is not None:
            return self.sample_actions(batch, num_steps=sample_steps, noise=noise)
        state = batch["state"]
        actions = batch["actions"]
        N, T_chunk, D_action = actions.shape

        if noise is None:
            noise = torch.randn_like(actions)
        if t is None:
            t = torch.rand(N, 1, 1, device=state.device, dtype=actions.dtype)

        if max_action_prefix > 0:
            apply_prefix = torch.rand(N, device=state.device) < prefix_conditioning_prob
            if "state_is_masked" in batch:
                apply_prefix = apply_prefix & ~batch["state_is_masked"].to(state.device)
            delay = torch.randint(0, max_action_prefix, (N,), device=state.device)
            delay = torch.where(apply_prefix, delay, torch.zeros_like(delay))
            prefix_mask = torch.arange(T_chunk, device=state.device)[None, :] < delay[:, None]
            prefix_mask_expanded = prefix_mask.unsqueeze(-1)
            t_per_pos = torch.where(prefix_mask_expanded, torch.zeros_like(t), t)
        else:
            prefix_mask_expanded = None
            t_per_pos = t

        x_t = (1 - t_per_pos) * actions + t_per_pos * noise
        if prefix_noise_scale > 0.0 and prefix_mask_expanded is not None:
            x_t = x_t + prefix_mask_expanded.to(x_t.dtype) * torch.randn_like(x_t) * prefix_noise_scale

        vision_tokens = self.build_vision_tokens(
            batch.get("images"), batch.get("camera_mask"), features=batch.get("vision_features")
        )
        task = self.encode_task(batch)
        embodiment = batch.get("embodiment_id")
        if prefix_mask_expanded is not None and compact_prefix_conditioning:
            sample_t = t[:, 0, 0]
            c = (self.compute_cond(state, task, sample_t, embodiment),
                 self.compute_cond(state, task, torch.zeros_like(sample_t), embodiment),
                 prefix_mask)
        else:
            t_cond = t_per_pos.squeeze(-1) if prefix_mask_expanded is not None else t[:, 0, 0]
            c = self.compute_cond(state, task, t_cond, embodiment)
        v_t = self.predict_velocity(x_t, c, vision_tokens)

        u_t = noise - actions
        valid = torch.ones_like(u_t)
        if prefix_mask_expanded is not None:
            valid = valid * (~prefix_mask_expanded).to(u_t.dtype)
        if "action_mask" in batch:
            valid = valid * batch["action_mask"].to(device=u_t.device, dtype=u_t.dtype)
        if valid.numel() and float(valid.min()) < 1:
            mse = ((u_t - v_t) ** 2 * valid).sum() / (valid.sum() + 1e-8)
        else:
            mse = F.mse_loss(u_t, v_t)
        return mse

    def _sample_setup(self, batch, noise):
        state = batch["state"]
        dtype = self.y_embedder.weight.dtype
        if noise is None:
            noise = torch.randn(
                state.shape[0],
                self.chunk_length,
                self.action_dim,
                device=state.device,
                dtype=dtype,
            )
        x_t = noise.to(device=state.device, dtype=dtype)
        vision = self.build_vision_tokens(
            batch.get("images"), batch.get("camera_mask"), features=batch.get("vision_features")
        )
        return state, dtype, x_t, vision, self.encode_task(batch), batch.get("embodiment_id")

    @torch.no_grad()
    def sample_actions(self, batch, num_steps=10, noise=None):
        """Euler flow integration from noise to actions.
        Vision tokens are computed once and reused across steps."""
        state, dtype, x_t, vision, task, embodiment_id = self._sample_setup(batch, noise)
        dt = -1.0 / num_steps
        b = state.shape[0]
        for i in range(num_steps):
            t = torch.full((b,), 1.0 + i * dt, device=state.device, dtype=dtype)
            x_t = x_t + self.predict_velocity(
                x_t, self.compute_cond(state, task, t, embodiment_id), vision
            ) * dt
        return x_t

    @torch.no_grad()
    def sample_actions_rtc(self, batch, action_prefix, prefix_length: int, num_steps=10, noise=None):
        """Euler sampling with per-position action-prefix conditioning."""
        state, dtype, x_t, vision, task, embodiment_id = self._sample_setup(batch, noise)
        action_prefix = action_prefix.to(device=state.device, dtype=dtype)
        prefix_pos = torch.arange(self.chunk_length, device=state.device) < prefix_length
        prefix_mask = prefix_pos.view(1, self.chunk_length, 1).expand_as(x_t)
        prefix_t_mask = prefix_pos.view(1, self.chunk_length).expand(state.shape[0], self.chunk_length)
        x_t = torch.where(prefix_mask, action_prefix, x_t)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full(
                (state.shape[0], self.chunk_length),
                1.0 + i * dt,
                device=state.device,
                dtype=dtype,
            )
            t = torch.where(prefix_t_mask, torch.zeros_like(t), t)
            x_t = x_t + self.predict_velocity(
                x_t, self.compute_cond(state, task, t, embodiment_id), vision
            ) * dt
            x_t = torch.where(prefix_mask, action_prefix, x_t)
        return x_t


_VISION_POOL_KEY_PREFIXES = (
    ("apool_queries.", "vision_pool.queries."),
    ("apool.", "vision_pool.apool."),
    ("vision_tokens_proj.", "vision_pool.proj."),
    ("vision_camera_embed.", "vision_pool.camera_embed."),
)


def _remap_vision_pool_keys(sd: dict) -> dict:
    """Map pre-``VisionTokenPool`` checkpoint keys onto the nested module."""
    if any(k.startswith("vision_pool.") for k in sd):
        return sd
    remapped = {}
    for key, value in sd.items():
        new_key = key
        for old, new in _VISION_POOL_KEY_PREFIXES:
            if key.startswith(old):
                new_key = new + key[len(old) :]
                break
        remapped[new_key] = value
    return remapped


def load_pretrained(model, ckpt_path):
    """Load a model-only checkpoint (prefixes stripped)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    if "task_to_hidden.weight" in sd and "language_pool_proj.proj.weight" not in sd:
        sd["language_pool_proj.proj.weight"] = sd.pop("task_to_hidden.weight")
        sd["language_pool_proj.proj.bias"] = sd.pop("task_to_hidden.bias")
    sd = _remap_vision_pool_keys(sd)
    load_into(model, sd, name="checkpoint")
    return ckpt
