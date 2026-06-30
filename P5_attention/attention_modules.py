"""
Attention modules for LogiK-Net (ToN major revision, concerns 1.1 / 2.3).

All models here expose the SAME interface as `utils.MLP` and
`utils.MultiKANModel`:

    forward(x, training=False) -> logits  (shape [B, n_classes])

so they can be dropped straight into `LogitsToPredicate` / `ltn.Predicate`
without touching the LTN backward path.

Key design choice (answers the reviewers' "spatial input" objection):
we do NOT imagify the tabular flow features. Instead each scalar feature is
treated as a token (feature tokenization, as in TabTransformer / FT-Transformer),
and self-attention is applied ACROSS features. This is dimension-agnostic and
gives an attention matrix that is directly interpretable as feature-feature
interaction importance.

Three classes:
  * FeatureAttentionEncoder  - tokenize features -> self-attention -> refined
                               representation. Can return SAME-dim features
                               (for the KAN front-end) or a pooled embedding.
  * AttentionKANModel        - FeatureAttentionEncoder (same-dim) -> KAN.
                               This is the LogiK-Net "+attention encoder" variant.
  * TransformerClassifier    - FeatureAttentionEncoder (pooled) -> MLP head.
                               Standalone Transformer baseline (the missing
                               HiViT-style comparison), no KAN, no logic.

Author note: written to run on the cluster where CIC_IoMT lives; this file has
no external deps beyond torch.
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Self-attention that also returns the attention weights (for interpretability)
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product MHSA, batch_first, returns attn weights.

    We implement it directly (instead of nn.MultiheadAttention) so the
    per-head attention maps are easy to pull out for the interpretability
    figures the reviewers asked for.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # [3, B, H, T, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                          # [B, H, T, d_head]
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.proj(out)
        # attn averaged over heads -> [B, T, T] for visualization
        return out, attn.mean(dim=1)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block: MHSA + FFN with residuals."""

    def __init__(self, d_model, n_heads, ff_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, attn = self.attn(self.norm1(x))
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn


# --------------------------------------------------------------------------- #
#  Feature-tokenizing attention encoder
# --------------------------------------------------------------------------- #
class FeatureAttentionEncoder(nn.Module):
    """Tokenize each scalar feature and run self-attention across features.

    Args:
        in_features : number of input features F (e.g. 9 for the 4_9 setup).
        d_model     : per-feature embedding dim.
        n_heads     : attention heads.
        n_layers    : number of transformer blocks.
        dropout     : dropout prob.
        out_mode    : 'same_dim' -> returns refined [B, F] (KAN front-end);
                      'pooled'   -> returns [CLS] embedding [B, d_model].
        residual    : (same_dim only) add input as residual to the refined
                      features so the encoder starts near identity.
    """

    def __init__(self, in_features, d_model=32, n_heads=4, n_layers=2,
                 dropout=0.1, out_mode="same_dim", residual=True):
        super().__init__()
        assert out_mode in ("same_dim", "pooled")
        self.in_features = in_features
        self.d_model = d_model
        self.out_mode = out_mode
        self.residual = residual

        # per-feature linear embedding: value * w_f + b_f  -> [d_model]
        self.feat_weight = nn.Parameter(torch.randn(in_features, d_model) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(in_features, d_model))
        # learned feature-identity ("column") embedding, acts like positional emb
        self.col_embed = nn.Parameter(torch.randn(in_features, d_model) * 0.02)

        if out_mode == "pooled":
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        if out_mode == "same_dim":
            # project each token back to a scalar -> [B, F]
            self.to_scalar = nn.Linear(d_model, 1)
            self.out_norm = nn.LayerNorm(in_features)

        self._last_attn = None  # cache last-layer attention for interpretability

    def tokenize(self, x):
        # x: [B, F] -> tokens [B, F, d_model]
        tokens = x.unsqueeze(-1) * self.feat_weight.unsqueeze(0) \
            + self.feat_bias.unsqueeze(0)
        tokens = tokens + self.col_embed.unsqueeze(0)
        return tokens

    def forward(self, x, training=False):
        tokens = self.tokenize(x)                      # [B, F, d]
        if self.out_mode == "pooled":
            cls = self.cls.expand(x.size(0), -1, -1)   # [B, 1, d]
            tokens = torch.cat([cls, tokens], dim=1)   # [B, F+1, d]

        attn_maps = []
        h = tokens
        for blk in self.blocks:
            h, attn = blk(h)
            attn_maps.append(attn)
        h = self.norm(h)
        self._last_attn = attn_maps[-1].detach()

        if self.out_mode == "pooled":
            return h[:, 0]                             # [B, d]  (CLS)

        # same_dim: token -> scalar
        refined = self.to_scalar(h).squeeze(-1)        # [B, F]
        if self.residual:
            refined = refined + x
        refined = self.out_norm(refined)
        return refined

    def last_attention(self):
        """Return last-layer attention map [B, T, T] from the most recent
        forward pass (feature-feature interaction importance)."""
        return self._last_attn


# --------------------------------------------------------------------------- #
#  LogiK-Net + attention encoder  (encoder -> KAN)
# --------------------------------------------------------------------------- #
class AttentionKANModel(nn.Module):
    """FeatureAttentionEncoder (same-dim) followed by a KAN logits model.

    `kan_model` is anything with forward(x, training=False) -> logits, i.e.
    an instance of utils.MultiKANModel. Because the encoder is same-dim, the
    KAN width is unchanged.
    """

    def __init__(self, in_features, kan_model, d_model=32, n_heads=4,
                 n_layers=2, dropout=0.1, residual=True):
        super().__init__()
        self.encoder = FeatureAttentionEncoder(
            in_features, d_model, n_heads, n_layers, dropout,
            out_mode="same_dim", residual=residual)
        self.kan = kan_model

    def forward(self, x, training=False):
        refined = self.encoder(x, training=training)
        return self.kan(refined, training=training)

    def last_attention(self):
        return self.encoder.last_attention()


# --------------------------------------------------------------------------- #
#  Standalone Transformer baseline (no KAN, no logic) -- the HiViT-style baseline
# --------------------------------------------------------------------------- #
class TransformerClassifier(nn.Module):
    """Feature-tokenized Transformer encoder + MLP head -> logits.

    Drop-in replacement for utils.MLP as a *baseline*. Same interface, so it
    trains through the identical LTN loop when wrapped in LogitsToPredicate,
    OR can be trained with plain cross-entropy as a pure baseline.
    """

    def __init__(self, in_features, n_classes, d_model=32, n_heads=4,
                 n_layers=2, dropout=0.1, head_hidden=(64, 32)):
        super().__init__()
        self.encoder = FeatureAttentionEncoder(
            in_features, d_model, n_heads, n_layers, dropout,
            out_mode="pooled")
        layers, prev = [], d_model
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x, training=False):
        emb = self.encoder(x, training=training)   # [B, d_model]
        return self.head(emb)                       # [B, n_classes]

    def last_attention(self):
        return self.encoder.last_attention()


if __name__ == "__main__":
    # quick shape check
    B, F, C = 8, 9, 13
    x = torch.randn(B, F)

    enc = FeatureAttentionEncoder(F, out_mode="same_dim")
    print("same_dim out:", enc(x).shape, "attn:", enc.last_attention().shape)

    tf = TransformerClassifier(F, C)
    print("transformer logits:", tf(x).shape)
