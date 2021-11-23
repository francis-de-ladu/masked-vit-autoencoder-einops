import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from utils.data import load_dataset
from vit import ViT

if __name__ == '__main__':
    model = ViT(
        in_channels=1,
        img_size=28,
        patch_size=4,
        emb_size=128,
        depth=4,
        expansion=4,
        n_heads=4,
        mask_ratio=.75,
    )

    # set seed and get splits
    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset(batch_size=1024)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    epochs = 10

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=lr)

    epoch_losses = []
    best_epoch = {'epoch': 0, 'loss': np.inf}
    for epoch in tqdm(range(epochs)):
        model.train()
        batch_losses = []
        for img, _ in train_loader:
            # send inputs to device
            img = img.to(device)

            # forward pass
            reconst_loss, reconst = model(img)
            print(reconst.shape)

            # backward pass
            optimizer.zero_grad()
            reconst_loss.backward()
            optimizer.step()

            # append the loss for this batch
            batch_losses.append(reconst_loss.detach().cpu().numpy())

        # compute train loss for the epoch
        mean_epoch_loss = np.sum(batch_losses) / len(train_loader.dataset)
        epoch_losses.append(mean_epoch_loss)

        # # compute eval loss for the epoch
        # eval_loss = evaluate(model, device, loss_fn, eval_loader)
        # best_epoch = save_if_best(model, epoch, eval_loss, best_epoch)

        print(f'Epoch {epoch + 1}:')
        print(f'Train loss: {mean_epoch_loss}')
        # print(f'Eval loss: {eval_loss}')

        # plot_reconst(model, device, eval_loader, epoch, noise=noise)
