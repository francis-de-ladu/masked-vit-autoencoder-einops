import numpy as np
from utils.misc import get_device


def evaluate(model, device, loss_fn, eval_loader):
    device = get_device()
    model.eval()
    eval_losses = []
    for x, _ in eval_loader:
        # send inputs to device
        x = x.to(device)

        # forward pass
        reconst = model(x)
        loss = loss_fn(x, reconst)

        # append the loss for this batch
        eval_losses.append(loss.detach().cpu().numpy())

    mean_eval_loss = np.sum(eval_losses) / len(eval_loader.dataset)
    return mean_eval_loss
