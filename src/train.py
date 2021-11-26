import os

import numpy as np
import torch
from config import config_base
from mae import MaskedAE
from torch import nn, optim
from tqdm import tqdm
from utils.data import load_dataset
from utils.test import evaluate
from utils.visuals import plot_reconst
from vit import ViT


def loss_fn_wrapper(loss_fn, device):
    def wrapped_loss_fn(masked_reconst, patches, masked_ids, unmasked_ids):
        batch_range = torch.arange(patches.size(0), device=device)[:, None]
        masked_patches = patches[batch_range, masked_ids]
        return loss_fn(masked_reconst, masked_patches)
    return wrapped_loss_fn


def save_if_best(model, epoch, eval_loss, best_epoch, save_dir='../models'):
    if eval_loss < best_epoch['loss']:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir.rstrip("/")}/{model.name}.model'
        torch.save(model.state_dict(), save_path)
        best_epoch.update({'epoch': epoch, 'loss': eval_loss})
    return best_epoch


if __name__ == '__main__':
    # set seed and get splits
    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset(batch_size=128)

    # get input dimensions
    in_channels, image_size, _ = train_loader.dataset[0][0].shape

    # instantiate vit encoder
    vit = ViT(
        in_channels,
        image_size,
        patch_size=4,
        n_classes=None,
        **config_base
    )

    # instantiate masked autoencoder
    model = MaskedAE(
        vit,
        mask_ratio=.75,
        decoder_dim=config_base['d_model'] // 4,
        decoder_depth=config_base['depth'] // 4,
        decoder_heads=config_base['n_heads']
    )

    # send model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 100
    lr = 1e-3

    loss_fn = loss_fn_wrapper(nn.MSELoss(reduction='sum'), device=device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=lr)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    epoch_losses = []
    best_epoch = {'epoch': 0, 'loss': np.inf}
    for epoch in tqdm(range(epochs)):
        model.train()
        batch_losses = []
        for img, _ in train_loader:
            # send inputs to device
            img = img.to(device)

            # forward pass
            with torch.autocast(device_type=device.type):
                reconst_loss = loss_fn(*model(img))

            # backward pass
            scaler.scale(reconst_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # reset gradients
            optimizer.zero_grad(set_to_none=True)

            # append the loss for this batch
            batch_losses.append(reconst_loss.detach().cpu().numpy())

        # compute train loss for the epoch
        mean_epoch_loss = np.sum(batch_losses) / len(train_loader.dataset)
        epoch_losses.append(mean_epoch_loss)

        # compute eval loss for the epoch
        eval_loss = evaluate(model, loss_fn, valid_loader)
        best_epoch = save_if_best(model, epoch, eval_loss, best_epoch)

        print(f'Epoch {epoch + 1}:')
        print(f'Train loss: {mean_epoch_loss}')
        print(f'Eval loss: {eval_loss}')

        saved_seed = torch.get_rng_state()
        torch.manual_seed(42)
        plot_reconst(model, valid_loader, epoch, noise=0)
        torch.set_rng_state(saved_seed)
