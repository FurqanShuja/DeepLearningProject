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

def evaluate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                losses = sum(sum(loss for loss in d.values()) for d in loss_dict)
            else:
                raise TypeError(f"Expected loss_dict to be a dict or list of dicts, but got {type(loss_dict)}")

            total_loss += losses.item()

            # Assume outputs and targets have the same length
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                target_boxes = target['boxes']
                target_labels = target['labels']

                correct = (pred_labels == target_labels).sum().item()
                total_correct += correct
                total_samples += target_labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            print(f"NaN or Inf encountered in epoch {epoch + 1} train loss.")
            continue

        train_losses.append(train_loss)

        val_loss, accuracy = evaluate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}")

    return train_losses, val_losses, accuracies
