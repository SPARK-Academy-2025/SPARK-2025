import torch
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Training Function
    """
    model.train()
    running_loss = 0.0
    if hasattr(criterion, 'loss_details'):
        criterion.set_phase("train")

    for _, (x, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

        if hasattr(criterion, 'loss_details'):
            loss_logging = criterion.loss_details.copy()
        else:
            loss_logging = {"train/loss": loss.item()}

    return running_loss / len(loader.dataset), loss_logging

def eval_epoch(model, loader, criterion, device, epoch):
    """
    Validation Function
    """
    model.eval()
    running_loss = 0.0
    if hasattr(criterion, 'loss_details'):
        criterion.set_phase("val")

    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)

        if hasattr(criterion, 'loss_details'):
            loss_logging = criterion.loss_details.copy()
        else:
            loss_logging = {"val/loss": loss.item()}
            
    return running_loss / len(loader.dataset), loss_logging
