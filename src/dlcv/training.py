import torch
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision.ops import nms

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Training")):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, list):
            losses = sum(sum(loss for loss in d.values()) for d in loss_dict)
        else:
            raise TypeError(f"Expected loss_dict to be a dict or list of dicts, but got {type(loss_dict)}")

        if torch.isnan(losses) or torch.isinf(losses):
            print(f"NaN or Inf detected in loss computation. Skipping this batch.")
            continue

        losses.backward()


        optimizer.step()
        optimizer.zero_grad()

        total_loss += losses.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def train_and_evaluate_model(model, train_loader, optimizer, num_epochs, device, scheduler=None):
    train_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            print(f"NaN or Inf encountered in epoch {epoch + 1} train loss.")
            continue

        train_losses.append(train_loss)

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss = {train_loss:.4f}")

    return train_losses
