import numpy as np


def evaluate(model, loss_fn, eval_loader):
    model.eval()
    device = next(model.parameters()).device

    eval_losses = []
    for x, _ in eval_loader:
        # send inputs to device
        x = x.to(device)

        # forward pass
        loss = loss_fn(*model(x))

        # append the loss for this batch
        eval_losses.append(loss.detach().cpu().numpy())

    mean_eval_loss = np.sum(eval_losses) / len(eval_loader.dataset)
    return mean_eval_loss
