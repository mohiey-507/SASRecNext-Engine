from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from torch import Tensor


class RMSNorm(nn.Module):
    """RMSNorm removes mean-centering for faster execution."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """Swish Gated Linear Unit Feed-Forward Network."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_projected = self.w1(x)
        x_val, gate = x_projected.chunk(2, dim=-1)
        activated = self.dropout(F.silu(gate) * x_val)
        return self.dropout(self.w2(activated))  # type: ignore[no-any-return]


class RoPE(nn.Module):
    """Rotary Position Embeddings with dynamic dtype and device dispatching."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int = 200) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]

        # Match device and precision dynamically for autocast / mixed-precision compatibility

        cos = self.cos_cached[:, :seq_len, ...].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, :seq_len, ...].to(device=q.device, dtype=q.dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            half_dim = x.shape[-1] // 2
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat((-x2, x1), dim=-1)

        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        return q_rope, k_rope


class RoPESelfAttention(nn.Module):
    """Multi-Head Self Attention strictly using RoPE and bias=False."""

    def __init__(self, d_model: int, nhead: int, max_seq_len: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.nhead, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch_size, seq_len, H, head_dim)

        q, k = self.rope(q, k)

        # Reshape to (B, H, L, head_dim) layout expected by SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

        return self.dropout(self.out_proj(out))  # type: ignore[no-any-return]


class TransformerLayer(nn.Module):
    """Pre-Norm Transformer Block using RMSNorm, RoPE Attention, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_dropout: float,
        ffn_dim: int,
        ffn_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = RoPESelfAttention(
            d_model=d_model,
            nhead=n_heads,
            max_seq_len=max_seq_len,
            dropout=attn_dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=ffn_dropout,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), attention_mask)
        h = x + attn_out
        out = h + self.ffn(self.ffn_norm(h))
        return out  # type: ignore[no-any-return]


class StandardAttentionFF(nn.Module):
    """PointWise Feed Forward used in Meta's SASRec baseline."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation_fn: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        assert activation_fn in ("relu", "gelu"), f"Invalid activation_fn {activation_fn}"

        self._conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.GELU() if activation_fn == "gelu" else nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=embedding_dim,
                kernel_size=1,
            ),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Conv1D requires (B, D, N)
        return self._conv1d(inputs.transpose(-1, -2)).transpose(-1, -2) + inputs  # type: ignore[no-any-return]


class SASRecSelfAttention(nn.Module):
    """Multi-Head Self Attention matching Meta's baseline but accepting our (B, 1, L, L) mask."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        import torch.nn.functional as F

        batch_size, seq_len, d_model = query.shape

        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

        return self.out_proj(out)  # type: ignore[no-any-return]


class SASRecTransformerLayer(nn.Module):
    """SASRec block matching Meta's exact layer normalizations and residual paths."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        ffn_dim: int,
        ffn_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.attn = SASRecSelfAttention(
            d_model=d_model,
            nhead=n_heads,
            dropout=attn_dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.ffn = StandardAttentionFF(
            embedding_dim=d_model,
            hidden_dim=ffn_dim,
            activation_fn="relu",
            dropout_rate=ffn_dropout,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Query is LayerNorm(x), Key and Value are x
        q = self.attn_norm(x)
        attn_out = self.attn(
            query=q,
            key=x,
            value=x,
            attention_mask=attention_mask,
        )
        # Residual adds normalized Q and attention output
        mid = q + attn_out

        # Second LayerNorm
        mid_norm = self.ffn_norm(mid)

        # FFN internally adds its input (which is mid_norm) to its output
        out = self.ffn(mid_norm)

        return out  # type: ignore[no-any-return]
