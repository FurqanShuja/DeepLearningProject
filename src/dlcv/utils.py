import csv
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def adjust_learning_rate(optimizer, epoch, cfg):
    if epoch < cfg.TRAIN.WARMUP_EPOCHS:
        lr = cfg.TRAIN.BASE_LR * (cfg.TRAIN.WARMUP_FACTOR + (1 - cfg.TRAIN.WARMUP_FACTOR) * epoch / cfg.TRAIN.WARMUP_EPOCHS)
    else:
        lr = cfg.TRAIN.BASE_LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Adjusted learning rate to: {lr}")


def denormalize(tensor, mean, std):
    tensor = tensor.clone()  # Clone the tensor to avoid modifying the original
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_image_with_boxes(image, boxes, labels, scores, file_path):
    image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC format
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Convert to uint8 format for proper image saving

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = plt.Rectangle((x_min, y_min), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')  # Turn off the axis
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)  # Save with a higher DPI and without extra padding
    plt.show()  # Display the plot on the screen
    plt.close(fig)  # Close the figure to avoid memory issues
    
def custom_collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])

    # Stack images (assumes images are tensors of same shape)
    images = torch.stack(images, dim=0)
    
    return images, targets

def json_serializable(obj):
    if isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postfix .pth
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model's state_dict
    torch.save(model.state_dict(), path + ".pth")

def get_transforms(train=True, image_height=3509, image_width=2480, horizontal_flip_prob=0, rotation_degrees=0):
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.RandomHorizontalFlip(horizontal_flip_prob),
            transforms.RandomRotation(rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def write_results_to_csv(file_path, train_losses):
    """
    Writes the training losses and evaluation metrics to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        metrics_list (list): List of dictionaries containing evaluation metrics.
    """
    # Add .csv extension to the file path
    file_path_with_extension = file_path + ".csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path_with_extension), exist_ok=True)
    
    # Check if file already exists to append or write new headers
    file_exists = os.path.exists(file_path_with_extension)
    
    # Write results to CSV
    with open(file_path_with_extension, mode='w+', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch]])

