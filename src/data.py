import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.img_size = img_size
        self.augment = augment

        base_aug = [A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST)]
        if augment:
            trans = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
            base_aug += trans

        base_aug += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
        self.transform = A.Compose(base_aug)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # ensure mask is 0/1 float tensor with channel dim
        mask = (mask.float().unsqueeze(0) / 255.0)
        return image.float(), mask.float()
