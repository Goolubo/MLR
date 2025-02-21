import torch
import torchvision.transforms.functional as F
import random

class RandomNoise:
    """
    在一定概率下为图像添加高斯噪声。
    """
    def __init__(self, prob=0.5, mean=0.0, std=0.1):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() < self.prob:
            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)  # PIL -> Tensor [0,1]
            noise = torch.randn_like(img) * self.std + self.mean
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)
            img = F.to_pil_image(img)  # 转回 PIL
        return img
