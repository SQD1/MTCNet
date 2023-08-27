from torch.utils.data import Dataset
import glob
import torch
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def resize(label):
    label = label / 255  # 0-1
    label = label.reshape([1, label.shape[0], label.shape[1]])
    label = np.concatenate((1 - label, label), axis=0)  # 类别数为2   [2, H, W]
    return label

# Dataset
# WHU256 label: 0 1
class WHUDataset(Dataset):
    def __init__(self, root, transform=transform, reverse=None, two_channel=None):
        # image
        T1_image_path = glob.glob(root + '/before' + '/*.tif')
        T2_image_path = glob.glob(root + '/after' + '/*.tif')
        T1_label_path = glob.glob(root + '/beforeMask' + '/*.png')
        T2_label_path = glob.glob(root + '/afterMask' + '/*.png')
        label_path = glob.glob(root + '/changeMask' + '/*.png')

        T1_image_path.sort()
        T2_image_path.sort()
        T1_label_path.sort()
        T2_label_path.sort()
        label_path.sort()

        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.T1_label_path = T1_label_path
        self.T2_label_path = T2_label_path
        self.label_path = label_path

        self.transform = transform
        self.reverse = reverse
        self.two_channel = two_channel

    def __getitem__(self, idx):

        image1 = np.array(Image.open(self.T1_image_path[idx]))
        image2 = np.array(Image.open(self.T2_image_path[idx]))

        T1_label = cv.imread(self.T1_label_path[idx], 0)
        T2_label = cv.imread(self.T2_label_path[idx], 0)
        label = cv.imread(self.label_path[idx], 0)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        if self.reverse and np.random.rand()<self.reverse:
            temp = image1
            image1 = image2
            image2 = temp
        if self.two_channel:
            label = resize(label)
            T1_label = resize(T1_label)
            T2_label = resize(T2_label)
        else:
            label = (label != 0).astype('uint8')
            T1_label = (T1_label != 0).astype('uint8')
            T2_label = (T2_label != 0).astype('uint8')

        label = torch.from_numpy(label).long()
        T1_label = torch.from_numpy(T1_label).long()
        T2_label = torch.from_numpy(T2_label).long()
        return image1, image2, T1_label, T2_label, label

    def __len__(self):
        return len(self.T1_image_path)

