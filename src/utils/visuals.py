import os

import matplotlib.pyplot as plt
import numpy as np
from utils.data import add_noise


def plot_reconst(model, device, eval_loader, epoch, noise, save_dir='../visuals'):
    model.eval()
    for orig, _ in eval_loader:
        orig = orig.to(device)
        if noise > 0:
            orig = add_noise(orig, noise)
        reconst = model(orig)
        break

    orig = orig[:, 0].detach().cpu().numpy()
    reconst = reconst[1][:, 0].detach().cpu().numpy()

    grid_size = np.asarray([8, 8])
    input_size = orig.shape[1:]
    image = np.ones(grid_size * input_size) / 2

    for i in range(grid_size[0]):
        for j in range(grid_size[1] // 2):
            x_start, x_end = 2 * j * input_size[1], 2 * (j + 1) * input_size[1]
            y_start, y_end = i * input_size[0], (i + 1) * input_size[0]
            x_mid = (x_start + x_end) // 2

            index = i * grid_size[1] // 2 + j
            image[y_start:y_end, x_start:x_mid] = orig[index]
            image[y_start:y_end, x_mid:x_end] = reconst[index]

    os.makedirs(save_dir, exist_ok=True)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.savefig(f'{save_dir.rstrip("/")}/epoch_{epoch+1}.png')
