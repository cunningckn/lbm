"""OpenAI CLIP ViT-B/32 text encoder."""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.config import ClipConfig
from lbm.models.attention import attention, split_qkv
from lbm.models.common import BFloat16Mixin
from lbm.models.weights import download_if_missing, env_paths, first_existing, load_into, local_checkpoint_candidates

SOT_TOKEN = "<|startoftext|>"
EOT_TOKEN = "<|endoftext|>"
_CLIP_TEXT_PREFIXES = (
    "token_embedding",
    "positional_embedding",
    "transformer.resblocks",
    "ln_final",
    "text_projection",
)


def task_name_to_prompt(task_name):
    """Convert task names like open_the_pen_caps to CLIP prompt text."""
    return " ".join(task_name.replace("-", " ").replace("_", " ").split())


def _load_clip_text_deps():
    try:
        import ftfy
        import regex
    except ImportError as exc:
        raise RuntimeError(
            "CLIP text embedding requires the 'ftfy' and 'regex' packages. "
            "Install lbm with its dependencies, or `uv sync` in this repo."
        ) from exc
    return ftfy, regex


def ensure_clip_text_assets(config: ClipConfig):
    """Download CLIP ViT-B/32 text assets if needed."""
    root = Path(config.cache_dir).expanduser()
    b32_path = root / config.model_name
    bpe_path = root / config.bpe_name
    download_if_missing(config.model_url, b32_path)
    download_if_missing(config.bpe_url, bpe_path)
    return b32_path, bpe_path


def resolve_clip_path() -> Path | None:
    cfg = ClipConfig()
    return first_existing(
        *env_paths("LBM_CLIP", "lbm_CLIP"),
        Path(cfg.cache_dir).expanduser() / cfg.model_name,
        *local_checkpoint_candidates("clip", cfg.model_name),
    )


def ensure_clip_weights() -> Path:
    path = resolve_clip_path()
    if path is not None:
        return path
    b32_path, _ = ensure_clip_text_assets(ClipConfig())
    return b32_path


def _clip_text_state(state_dict: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith(_CLIP_TEXT_PREFIXES)}


def _load_openai_clip_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache()
def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    return set(zip(word, word[1:]))


class CLIPBPETokenizer:
    """Small copy of OpenAI CLIP's BPE tokenizer, scoped to text encoding."""

    def __init__(self, bpe_path):
        _ftfy, regex = _load_clip_text_deps()
        self.ftfy = _ftfy
        self.regex = regex
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = [tuple(line.split()) for line in f.read().split("\n")[1 : 49152 - 256 - 2 + 1]]
        self.byte_encoder = _bytes_to_unicode()
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend([SOT_TOKEN, EOT_TOKEN])
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {SOT_TOKEN: SOT_TOKEN, EOT_TOKEN: EOT_TOKEN}
        self.pat = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|\'s|\'t|\'re|\'ve|\'m|\'ll|\'d|"
            r"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _basic_clean(self, text):
        return html.unescape(html.unescape(self.ftfy.fix_text(text))).strip()

    def _whitespace_clean(self, text):
        return self.regex.sub(r"\s+", " ", text).strip()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if word[j] == first and j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        text = self._whitespace_clean(self._basic_clean(text)).lower()
        tokens = []
        for token in self.regex.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(token).split(" "))
        return tokens


class CLIPQuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def _mha_self_attn(attn: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """Causal self-attn using ``attn``'s in/out projections and ``attention()``.

    ``x`` is ``(L, N, E)`` (CLIP / ``batch_first=False``). Matches
    ``nn.MultiheadAttention`` with an upper-triangular ``-inf`` mask.
    """
    seq, batch, embed = x.shape
    n_heads = attn.num_heads
    head_dim = embed // n_heads
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = split_qkv(qkv.permute(1, 0, 2), n_heads, head_dim)
    y = attention(q, k, v, causal=True)
    y = y.permute(2, 0, 1, 3).contiguous().view(seq, batch, embed)
    return F.linear(y, attn.out_proj.weight, attn.out_proj.bias)


class CLIPTextBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        self.mlp.add_module("gelu", CLIPQuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x):
        x = x + _mha_self_attn(self.attn, self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTextTower(nn.Module):
    """OpenAI CLIP ViT-B/32 text transformer (causal, QuickGELU, EOT @ projection)."""

    def __init__(
        self,
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(
            *[CLIPTextBlock(width, heads) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width**-0.5)

    def forward(self, text):
        x = self.token_embedding(text) + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer.resblocks(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)] @ self.text_projection


def load_clip(module: CLIPTextTower | CLIPLanguageEncoder | CLIPTextEmbedder, ckpt_path):
    """Load OpenAI CLIP ViT-B/32 text weights into a text tower."""
    text_keys = _clip_text_state(_load_openai_clip_state(Path(ckpt_path).expanduser()))
    target = module.model if isinstance(module, (CLIPLanguageEncoder, CLIPTextEmbedder)) else module
    return load_into(target, text_keys, name="CLIP text")


class CLIPTextEmbedder(BFloat16Mixin):
    """OpenAI CLIP ViT-B/32 text encoder that returns normalized 512-d vectors.

    Holds a CPU memo cache keyed by prompt so repeats skip BPE+transformer.
    """

    def __init__(self, config: ClipConfig, device="cpu"):
        b32_path, bpe_path = ensure_clip_text_assets(config)
        self.device = torch.device(device)
        self.bfloat16 = False
        self.tokenizer = CLIPBPETokenizer(bpe_path)
        self.model = CLIPTextTower().eval().to(self.device)
        load_clip(self, b32_path)
        self._cache = {}

    def _encode_tokens(self, context: torch.Tensor) -> torch.Tensor:
        return self._maybe_bf16(context, lambda: self.model(context))

    def tokenize(self, texts) -> tuple[torch.Tensor, torch.Tensor]:
        """CLIP BPE ids and pad mask, shape ``(B, 77)``."""
        if isinstance(texts, str):
            texts = [texts]
        ctx = self.model.context_length
        tokens = torch.zeros(len(texts), ctx, dtype=torch.long)
        mask = torch.zeros(len(texts), ctx, dtype=torch.float32)
        sot = self.tokenizer.encoder[SOT_TOKEN]
        eot = self.tokenizer.encoder[EOT_TOKEN]
        for i, text in enumerate(texts):
            token_ids = [sot, *self.tokenizer.encode(str(text)), eot]
            if len(token_ids) > ctx:
                token_ids = token_ids[: ctx - 1] + [eot]
            n = len(token_ids)
            tokens[i, :n] = torch.tensor(token_ids, dtype=torch.long)
            mask[i, :n] = 1.0
        return tokens, mask

    @torch.no_grad()
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        fresh = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if fresh:
            context = torch.zeros(
                len(fresh), self.model.context_length, dtype=torch.long, device=self.device
            )
            for i, text in enumerate(fresh):
                token_ids = [
                    self.tokenizer.encoder[SOT_TOKEN],
                    *self.tokenizer.encode(text),
                    self.tokenizer.encoder[EOT_TOKEN],
                ]
                if len(token_ids) > self.model.context_length:
                    raise RuntimeError(
                        f"Input {text!r} is too long for CLIP context length "
                        f"{self.model.context_length}"
                    )
                context[i, : len(token_ids)] = torch.tensor(
                    token_ids, dtype=torch.long, device=self.device
                )
            features = self._encode_tokens(context)
            features = features / features.norm(dim=-1, keepdim=True)
            for i, text in enumerate(fresh):
                self._cache[text] = features[i].cpu()
        out = torch.stack([self._cache[t] for t in texts], dim=0)
        return out.to(self.device)


class CLIPLanguageEncoder(BFloat16Mixin, nn.Module):
    """In-graph original CLIP ViT-B/32 text tower. ``forward(input_ids) -> (B, 512)``."""

    def __init__(self, config: ClipConfig | None = None):
        super().__init__()
        self.model = CLIPTextTower()
        self.out_dim = int(self.model.text_projection.shape[1])
        self.bfloat16 = False
        load_clip(self, ensure_clip_weights() if config is None else ensure_clip_text_assets(config)[0])

    def forward(self, input_ids, attention_mask=None):
        ctx = self.model.context_length
        if input_ids.shape[1] != ctx:
            padded = input_ids.new_zeros(input_ids.shape[0], ctx)
            n = min(input_ids.shape[1], ctx)
            padded[:, :n] = input_ids[:, :n]
            input_ids = padded
        features = self._maybe_bf16(input_ids, lambda: self.model(input_ids))
        return features / features.norm(dim=-1, keepdim=True)


def encode_clip_text(texts, config: ClipConfig, device="cpu"):
    """Encode exact prompt text with OpenAI CLIP ViT-B/32."""
    return CLIPTextEmbedder(config, device=device).encode(texts)


def encode_clip_task_name(task_names, config: ClipConfig, device="cpu"):
    """Encode task names after dash/underscore replacement."""
    if isinstance(task_names, str):
        task_names = [task_names]
    return encode_clip_text([task_name_to_prompt(t) for t in task_names], config, device=device)
