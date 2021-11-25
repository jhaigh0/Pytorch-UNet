import logging
import os

import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, imgDir, maskDir):
        self.imgPaths = []
        self.maskPaths = []

        # Get all img paths
        for path, folders, files in os.walk(imgDir):
            for file in files:
                self.imgPaths.append(os.path.join(path, file))

        # Get all mask paths
        for path, folders, files in os.walk(maskDir):
            for file in files:
                self.maskPaths.append(os.path.join(path, file))

        # Check number of images and masks are the same
        if len(self.imgPaths) != len(self.maskPaths):
            raise Exception("Different number of images and masks")
        logging.info(f'Creating dataset with {len(self.imgPaths)} examples')

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        return self.loadImage(self.imgPaths[idx]), self.loadMask(self.maskPaths[idx])

    @staticmethod
    def loadImage(path):
        image = read_image(path, ImageReadMode.GRAY)
        image = transforms.ConvertImageDtype(torch.float32)(image)
        return image

    @staticmethod
    def loadMask(path):
        mask = read_image(path, ImageReadMode.GRAY)
        mask = mask.squeeze().to(torch.long)
        return mask