import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class ViT(nn.Module):
    def __init__(self,
                 in_channels,
                 img_size,
                 patch_size,
                 emb_size=128,
                 depth=4,
                 mask_ratio=.75,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.depth = depth
        self.mask_ratio = mask_ratio

        self.name = 'mae-vit'
        self.n_patches = (img_size // patch_size)**2
        self.num_masked = int(mask_ratio * self.n_patches)

        self.enc_to_dec = nn.Linear(emb_size, emb_size // 4)
        self.mask_token = nn.Parameter(torch.randn(emb_size // 4))
        self.dec_pos_embed = nn.Embedding(self.n_patches, emb_size // 4)
        self.dec_embed_to_pixels = nn.Linear(emb_size // 4, patch_size**2)

        self.embed = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size,
            self.n_patches, self.num_masked)
        self.encoder = TransformerEncoder(
            depth, emb_size=emb_size, **kwargs)
        self.decoder = TransformerDecoder(
            depth // 4, emb_size=emb_size // 4, **kwargs)

    def forward(self, img):
        masked_patches, unmasked_patches, unmasked_embeds = self.embed(img)
        encoded_tokens = self.encoder(unmasked_embeds)
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        mask_tokens = repeat(
            self.mask_token, 'd -> b n d', b=img.size(0), n=self.num_masked)
        mask_tokens += self.dec_pos_embed(self.embed.masked_ids)

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[:, :self.num_masked]
        masked_reconst = self.dec_embed_to_pixels(mask_tokens)

        reconst_loss = F.mse_loss(
            masked_reconst, masked_patches, reduction='sum')

        batch_range = torch.arange(img.size(0), device=img.device)[:, None]
        reconst = torch.cat((masked_reconst, unmasked_patches), dim=1)
        # reconst = torch.cat((masked_patches, unmasked_patches), dim=1)

        # rand_ids = torch.rand(
        #     [img.size(0), self.n_patches], device=img.device).argsort(dim=-1)
        # reconst = reconst[batch_range, rand_ids]
        reconst = reconst[batch_range, self.embed.rand_ids]

        reconst = rearrange(
            reconst, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            p1=self.patch_size, p2=self.patch_size, h=int(self.n_patches**0.5),
        )

        return reconst_loss, reconst


# PATCH EMBEDDINGS

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size,
                 n_patches, num_masked):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.num_masked = num_masked

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d'),
        )
        self.positions = nn.Parameter(torch.randn(self.n_patches, emb_size))

    def forward(self, img):
        tokens = self.project(img) + self.positions
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                            p1=self.patch_size, p2=self.patch_size)

        self.rand_ids = torch.rand(
            *tokens.shape[:2], device=img.device).argsort(dim=-1)
        self.masked_ids = self.rand_ids[:, :self.num_masked]
        self.unmasked_ids = self.rand_ids[:, self.num_masked:]

        batch_range = torch.arange(tokens.size(0), device=img.device)[:, None]
        masked_patches = patches[batch_range, self.masked_ids]
        unmasked_patches = patches[batch_range, self.unmasked_ids]
        unmasked_embeds = tokens[batch_range, self.unmasked_ids]

        return masked_patches, unmasked_patches, unmasked_embeds


# TRANSFORMER ENCODER

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        blocks = [TransformerBlock(**kwargs) for _ in range(depth)]
        super().__init__(*blocks)


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


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion):
        super().__init__(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            # TODO: add dropout here
            nn.Linear(emb_size * expansion, emb_size),
        )


# TRANSFORMER DECODER

class TransformerDecoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        blocks = [TransformerBlock(**kwargs) for _ in range(depth)]
        super().__init__(*blocks)


# TRANSFORMER BLOCK

class TransformerBlock(nn.Module):
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
        x = self.attn_block(x) + x
        x = self.ff_block(x) + x
        return x
