from tqdm import tqdm
import time
import torch
from torch import optim
from torch.optim import lr_scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=200,
    val_early_stop=0.99,
):
    since = time.time()
    patience = 5
    best_loss = float("inf")
    counter = 0
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_losses = list()
        train_accuracy = list()
        lrs = list()
        for inputs, labels in tqdm(train_loader, total=len(train_loader)):
            model.train()

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            train_losses.append(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # Record and update lr
            lrs.append(get_learning_rate(optimizer))

        val_losses = list()
        val_accuracy = list()
        for inputs, labels in tqdm(val_loader, total=len(val_loader)):
            model.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.float())
            labels = labels
            loss = criterion(outputs.float(), labels.float())
            val_losses.append(loss)

        result = {}
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        result["val_loss"] = torch.stack(val_losses).mean().item()
        scheduler.step(result["val_loss"])

        print(
            f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}"
        )
        if result["val_loss"] < best_loss:
            best_loss = result["val_loss"]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    return model, result
