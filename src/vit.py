import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class ViT(nn.Module):
    def __init__(
        self, in_channels, image_size, patch_size, n_classes,
        d_model, depth, n_heads, expansion, dropout
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout

        patches = (image_size // patch_size)**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(in_channels * patch_size**2, d_model),
        )
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size),
        #     Rearrange('b d h w -> b (h w) d'),
        # )
        self.pos_embedding = nn.Parameter(torch.randn(patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.dropout_layer = nn.Dropout(dropout)

        self.transformer = Transformer(
            d_model, depth, n_heads, expansion, dropout)

        self.classify = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.Tanh(),
            nn.Linear(d_model * expansion, n_classes),
        ) if n_classes else nn.Identity()

    def forward(self, img):
        tokens = self.to_patch_embedding(img)
        batch, patches, _ = tokens.shape

        cls_tokens = repeat(self.cls_token, 'd -> b n d', b=batch, n=patches)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens += self.pos_embedding
        tokens = self.dropout_layer(tokens)

        latent = self.transformer(tokens)

        out = self.classify(latent[:, 0])
        return out


# TRANSFORMER

class Transformer(nn.Sequential):
    def __init__(self, d_model, depth, n_heads, expansion, dropout):
        blocks = [TransformerBlock(d_model, n_heads, expansion, dropout)
                  for _ in range(depth)]
        super().__init__(*blocks)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, expansion, dropout):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion

        self.attn_block = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, n_heads),
            nn.Dropout(dropout),
        )

        self.ff_block = nn.Sequential(
            nn.LayerNorm(d_model),
            FeedForward(d_model, expansion, dropout),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.attn_block(x) + x
        x = self.ff_block(x) + x
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scaling = (d_model // n_heads)**(-0.5)

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.project = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q, k, v = rearrange(
            self.qkv(x), 'b n (h d qkv) -> qkv b h n d', h=self.n_heads, qkv=3)
        attn = self.attention(q, k, v)
        out = self.project(attn)
        return out

    def attention(self, q, k, v):
        scores = torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scaling
        attn = torch.einsum('bhad, bhdv -> bhav', self.softmax(scores), v)
        return rearrange(attn, 'b h n d -> b n (h d)')


class FeedForward(nn.Sequential):
    def __init__(self, d_model, expansion, dropout):
        super().__init__(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
        )
