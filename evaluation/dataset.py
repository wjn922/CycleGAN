import glob
import os
import random
import time

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImagenetDataset(Dataset):
    def __init__(self, root, label='horse', transform=None):
        """
        label: horse, zebra, orange, apple
        """
        self.label = label
        self.transform = transform

        self.files = sorted(glob.glob(root + "/*.*"))

        print("The dataset root is {}".format(root))
        print("The dataset label is {}".format(label))
        print("Please note to check whether dataset and label are consistent by yourself.")

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.label == 'horse':
            label = int(339)
        elif self.label == 'zebra':
            label = int(340)
        elif self.label == 'apple':
            label = int(948)
        elif self.label == 'orange':
            label = int(950)
        else:
            raise NotImplementedError("Label should be one of : 'horse', 'zebra', 'apple', 'orange'")

        return image, torch.tensor(label).long()

    def __len__(self):
        return len(self.files)