import os
from itertools import chain

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from utils.data import add_noise


def plot_reconst(model, eval_loader, epoch, noise, save_dir='../visuals'):
    model.eval()
    device = next(model.parameters()).device

    PATCH_SIZE = model.encoder.patch_size
    PATCH_PER_AXIS = model.encoder.image_size // PATCH_SIZE

    for orig, _ in eval_loader:
        orig = orig.to(device)
        if noise > 0:
            orig = add_noise(orig, noise)

        masked_reconst, patches, masked_ids, unmasked_ids = model(orig)
        shuffled_ids = torch.cat((masked_ids, unmasked_ids), dim=1)

        batch_range = torch.arange(orig.size(0), device=device)[:, None]
        masked_patches = 0.5 * torch.ones_like(masked_reconst)
        unmasked_patches = patches[batch_range, unmasked_ids]

        input_patches = torch.cat((masked_patches, unmasked_patches), dim=1)
        reconst_patches = torch.cat((masked_reconst, unmasked_patches), dim=1)

        input_patches[batch_range, shuffled_ids.argsort(dim=-1)]
        reconst_patches[batch_range, shuffled_ids.argsort(dim=-1)]
        orig_patches = rearrange(
            orig, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
            s1=PATCH_SIZE, s2=PATCH_SIZE,
        )

        # input = rearrange(
        #     input_patches[batch_range, shuffled_ids.argsort(dim=-1)],
        #     'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        #     p1=model.encoder.patch_size, p2=model.encoder.patch_size,
        #     h=model.encoder.image_size // model.encoder.patch_size,
        # )
        # reconst = rearrange(
        #     reconst_patches[batch_range, shuffled_ids.argsort(dim=-1)],
        #     'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        #     p1=model.encoder.patch_size, p2=model.encoder.patch_size,
        #     h=model.encoder.image_size // model.encoder.patch_size,
        # )
        break

    # orig = orig.detach().cpu().numpy()
    # input = input.detach().cpu().numpy()
    # reconst = reconst.detach().cpu().numpy()

    # orig = rearrange(orig, 'b c h w -> b h w c')
    # input = rearrange(input, 'b c h w -> b h w c')
    # reconst = rearrange(reconst, 'b c h w -> b h w c')

    # b n d -> (3 b) n d ->
    lines, cols = 8, 3
    num_samples = lines * cols

    triplets = zip(input_patches, reconst_patches, orig_patches)
    images = list(chain.from_iterable(triplets))[:(3 * num_samples)]

    images = rearrange(
        images, '(i j n) (p1 p2) (s1 s2 c) -> (i p1 s1) (j n p2 s2) c',
        i=lines, j=cols, n=3,
        p1=PATCH_PER_AXIS, p2=PATCH_PER_AXIS,
        s1=PATCH_SIZE, s2=PATCH_SIZE)
    images = images.detach().cpu().numpy()

    # prepare visualization
    cmap = 'gray' if orig.shape[-1] == 1 else None
    plt.imshow(images, cmap=cmap, vmin=0, vmax=1)
    plt.title(f'Epoch {epoch+1}')
    plt.axis('off')

    # save visualization
    save_path = f'{save_dir.rstrip("/")}/epoch_{epoch+1}.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
