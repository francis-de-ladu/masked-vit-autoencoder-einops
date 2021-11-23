import os

import numpy as np
import torch
from tqdm import tqdm
from utils.data import add_noise
from utils.test import evaluate
from utils.visuals import plot_reconst


def train(model, device, loss_fn, optimizer, train_loader, eval_loader, epochs=10, noise=False):
    epoch_losses = []
    best_epoch = {'epoch': 0, 'loss': np.inf}
    for epoch in tqdm(range(epochs)):
        model.train()
        batch_losses = []
        for orig, _ in train_loader:
            # send inputs to device
            orig = orig.to(device)

            # forward pass
            if noise > 0:
                with_noise = add_noise(orig, noise)
                reconst = model(with_noise)
            else:
                reconst = model(orig)
            loss = loss_fn(orig, reconst)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append the loss for this batch
            batch_losses.append(loss.detach().cpu().numpy())

        # compute train loss for the epoch
        mean_epoch_loss = np.sum(batch_losses) / len(train_loader.dataset)
        epoch_losses.append(mean_epoch_loss)

        # compute eval loss for the epoch
        eval_loss = evaluate(model, device, loss_fn, eval_loader)
        best_epoch = save_if_best(model, epoch, eval_loss, best_epoch)

        print(f'Epoch {epoch + 1}:')
        print(f'Train loss: {mean_epoch_loss}')
        print(f'Eval loss: {eval_loss}')

        plot_reconst(model, device, eval_loader, epoch, noise=noise)


def save_if_best(model, epoch, eval_loss, best_epoch, save_dir='../models'):
    if eval_loss < best_epoch['loss']:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir.rstrip("/")}/{model.name}.model'
        torch.save(model.state_dict(), save_path)
        best_epoch.update({'epoch': epoch, 'loss': eval_loss})
    return best_epoch
