import torch 
import cv2 
import numpy as np
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(input_size, patch_size, cfg, shuffle=True, drop_last=True):
    dataloader = DataLoader(
        dataset=CustomImageFolderDataset(cfg.image_folder, input_size, patch_size), 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader 

    
class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self, image_folder, input_size, patch_size):
        super(CustomImageFolderDataset, self).__init__(image_folder)
        self.root = image_folder 
        self.input_size = input_size
        self.patch_size = patch_size

        self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)
        ]), p=0.5), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_bgr = cv2.imread(path)

        if np.random.random() < 0.5:
            img_shape = img_bgr.shape 
            side_ratio = np.random.uniform(0.3, 0.7)
            new_shape = (int(side_ratio * img_shape[1]), int(side_ratio * img_shape[0]))
            interpolation = np.random.choice(
                [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
            )
            img_bgr = cv2.resize(img_bgr, new_shape, interpolation=interpolation)

        img_bgr = cv2.resize(img_bgr, (self.input_size[1], self.input_size[0]))

        img = Image.fromarray(img_bgr.astype(np.uint8))
        sample = self.transform(img)

        H, W = self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size
        high = int(np.random.uniform(0.0, 0.5) * H * W)
        mask = np.hstack([np.zeros(H * W - high), np.ones(high)]).astype(bool)
        np.random.shuffle(mask)

        return sample, target, mask 
