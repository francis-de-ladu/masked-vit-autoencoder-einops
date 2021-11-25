import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from utils.data import add_noise


def plot_reconst(model, eval_loader, epoch, noise, save_dir='../visuals'):
    model.eval()
    device = next(model.parameters()).device

    for orig, _ in eval_loader:
        orig = orig.to(device)
        if noise > 0:
            orig = add_noise(orig, noise)

        masked_reconst, patches, masked_ids, unmasked_ids = model(orig)

        batch_range = torch.arange(orig.size(0), device=device)[:, None]
        unmasked_patches = patches[batch_range, unmasked_ids]

        reconst_patches = torch.cat((masked_reconst, unmasked_patches), dim=1)
        reconst_ids = torch.cat((masked_ids, unmasked_ids), dim=1)

        reconst = reconst_patches[batch_range, reconst_ids.argsort(dim=-1)]
        reconst = rearrange(
            reconst,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            p1=model.encoder.patch_size, p2=model.encoder.patch_size,
            h=model.encoder.image_size // model.encoder.patch_size,
        )
        break

    orig = orig.detach().cpu().numpy()
    reconst = reconst.detach().cpu().numpy()

    orig = rearrange(orig, 'b c h w -> b h w c')
    reconst = rearrange(reconst, 'b c h w -> b h w c')

    grid_size = np.asarray([8, 8, 1])
    input_size = orig.shape[1:]
    image = np.ones(grid_size * input_size) / 2

    for i in range(grid_size[1]):
        for j in range(grid_size[0] // 2):
            x_beg, x_end = 2 * j * input_size[0], 2 * (j + 1) * input_size[0]
            y_beg, y_end = i * input_size[1], (i + 1) * input_size[1]
            x_mid = (x_beg + x_end) // 2

            index = i * grid_size[1] // 2 + j
            image[y_beg:y_end, x_beg:x_mid] = orig[index]
            image[y_beg:y_end, x_mid:x_end] = reconst[index]

    os.makedirs(save_dir, exist_ok=True)
    # plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.imshow(image, vmin=0, vmax=1)
    plt.savefig(f'{save_dir.rstrip("/")}/epoch_{epoch+1}.png')
