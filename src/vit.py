import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce, Repeat
from torch import nn


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, depth, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.depth = depth

        self.embed = PatchEmbedding(in_channels, patch_size, emb_size)
        self.encoder = TransformerEncoder(depth, **kwargs)
        self.decoder = TransformerDecoder(depth, **kwargs)

    def forward(self, img):
        patches = self.embed(img)
        encoded = self.encoder(patches)
        decoded = self.decoder(encoded)
        return decoded


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # TODO: add cls_token
        # TODO: add positional embedding

    def forward(self, x):
        x = self.embed(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, emb_size, expansion, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.expansion = expansion

        self.attn_block = nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, **kwargs),
            # TODO: dropout here
        )

        self.ff_block = nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=expansion),
            # TODO: dropout here
        )

    def forward(self, x):
        x = x + self.attn_block(x)
        x = x + self.ff_block(x)
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion):
        super().__init__(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            # TODO: add dropout here
            nn.Linear(emb_size * expansion, emb_size),
        )


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        blocks = [EncoderBlock(**kwargs) for _ in range(depth)]
        super().__init__(*blocks)


class TransformerDecoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        blocks = [DecoderBlock(**kwargs) for _ in range(depth)]
        super().__init__(*blocks)


# b c h w -> b n (p1 p2 c) -> b n embed -> b n (h d) -> b h n d -> qkv b h n d

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scaling = (d_model // n_heads)**(-0.5)

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        q, k, v = rearrange(
            self.qkv(x), 'b n (h d qkv) -> qkv b h n d', h=self.n_heads, qkv=3)
        attn = self.attention(q, k, v)
        out = self.fc(attn)
        return out

    def attention(self, q, k, v):
        # attn = torch.softmax(q @ k * self.scaling, dim=-1) @ v
        # return rearrange(attn, 'b h n d -> b (n h) d')
        scores = torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scaling
        attn = torch.einsum('bhsd, bhvd -> bhsv', scores, v)
        return rearrange(attn, 'b h n d -> b (n h) d')
