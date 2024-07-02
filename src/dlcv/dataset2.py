import os
from torch.utils.data import Dataset
from PIL import Image

class Dataset2(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        images = []
        images_dir = os.path.join(self.root, 'images', self.split)
        for filename in os.listdir(images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(images_dir, filename))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        file_name = os.path.basename(image_path)
        return image, file_name
