import torch
from einops import repeat
from torch import nn
from vit import Transformer


class MaskedAE(nn.Module):
    def __init__(self, encoder, mask_ratio,
                 decoder_dim, decoder_depth, decoder_heads,
                 name='mae-vit2'):
        super().__init__()
        self.name = name
        self.mask_ratio = mask_ratio

        # encoder layers
        self.encoder = encoder
        self.to_patch, self.patch_embed = encoder.to_patch_embedding

        # encoder hyperparams
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        values_per_patch = self.patch_embed.weight.shape[-1]

        # decoder layers
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) \
            if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            decoder_dim, decoder_depth, decoder_heads,
            encoder.expansion, encoder.dropout)
        self.dec_pos_embed = nn.Embedding(num_patches, decoder_dim)
        self.dec_embed_to_pixels = nn.Linear(decoder_dim, values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, _ = patches.shape

        # encode patches and add positional embeddings
        tokens = self.patch_embed(patches) + self.encoder.pos_embedding[1:]

        # divide ids in those to be masked/unmasked
        num_masked = int(self.mask_ratio * num_patches)
        rand_ids = torch.rand(batch, num_patches, device=device).argsort(-1)
        masked_ids = rand_ids[:, :num_masked]
        unmasked_ids = rand_ids[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_ids]

        # # get the patches to be masked for the final reconstruction loss
        # masked_patches = patches[batch_range, masked_ids]

        # attend with vision transformer to unmasked tokens
        encoded_tokens = self.encoder.transformer(unmasked_tokens)

        # project encoder to decoder dimensions
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # get mask tokens and add positional embeddings based on masked ids
        mask_tokens = repeat(
            self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.dec_pos_embed(masked_ids)

        # attend with decoder to mask and unmasked tokens
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # project mask tokens to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        masked_reconst = self.dec_embed_to_pixels(mask_tokens)

        return torch.sigmoid(masked_reconst), patches, masked_ids, unmasked_ids
