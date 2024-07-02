import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch

class Dataset1(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images, self.annotations = self.load_data()

    def load_data(self):
        annotations_file = os.path.join(self.root, 'annotations', f'{self.split}.json')
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}
        annotations = {img_id: [] for img_id in images.keys()}

        for ann in data['annotations']:
            annotations[ann['image_id']].append(ann)

        return images, annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_data = self.images[img_id]
        annotations = self.annotations[img_id]
        
        image_path = os.path.join(self.root, 'images', self.split, img_data['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")  # Debugging statement
            return self.__getitem__((idx + 1) % len(self))  # Skip the missing image and try the next one
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        targets = self.prepare_targets(annotations, img_id)
        
        # Skip images with no annotations
        if len(targets['boxes']) == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        return image, targets

    
    def prepare_targets(self, annotations, img_id):
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            if bbox:
                x_min, y_min, width, height = bbox
                x_max, y_max = x_min + width, y_min + height
                # Ensure the coordinates are within the image bounds
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, self.images[img_id]['width'])
                y_max = min(y_max, self.images[img_id]['height'])
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return targets

