import random
import math
import torch
from torchvision import transforms
from PIL import Image, ImageOps

class CutPasteNormal:
    """
    基础 CutPaste 版本：从图像中随机裁一个patch，再粘贴到图像的其他随机位置上
    """
    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=0.3, transform=None):
        """
        :param area_ratio: (min_area, max_area)，裁剪patch相对于整图面积的占比区间
        :param aspect_ratio: 宽高比的下限值（逆为1/aspect_ratio），用于随机生成不同形状
        :param transform: 对裁剪下来的patch做进一步处理的transform，比如颜色抖动
        """
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.transform = transform

    def __call__(self, img):
        # 将输入转换成 PIL image
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        w, h = img.size

        # 随机生成裁剪区域大小
        area = w * h
        patch_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area

        # 随机生成宽高比
        log_ratio = (math.log(self.aspect_ratio), math.log(1.0 / self.aspect_ratio))
        aspect = math.exp(random.uniform(*log_ratio))

        patch_w = int(round(math.sqrt(patch_area * aspect)))
        patch_h = int(round(math.sqrt(patch_area / aspect)))

        # 在图像中随机选定裁剪区域位置
        x1 = random.randint(0, max(w - patch_w, 0))
        y1 = random.randint(0, max(h - patch_h, 0))
        box = (x1, y1, x1 + patch_w, y1 + patch_h)

        # 裁剪 patch
        patch = img.crop(box)
        if self.transform:
            patch = self.transform(patch)  # 对patch可做额外变换

        # 目标粘贴位置
        x2 = random.randint(0, max(w - patch_w, 0))
        y2 = random.randint(0, max(h - patch_h, 0))

        # 把 patch 粘贴回图像
        augmented = img.copy()
        augmented.paste(patch, (x2, y2, x2 + patch_w, y2 + patch_h))

        return augmented


class CutPasteScar:
    """
    Scar 版本：在裁剪后，模拟一个狭长“伤口形”patch
    """
    def __init__(self, width=[0.02, 0.15], height_ratio=[0.1, 0.3], rotation=[-45, 45], transform=None):
        """
        :param width: 裁剪patch相对于图像宽度的比例范围 (min, max)
        :param height_ratio: patch 高度与宽度的比率范围
        :param rotation: 对patch执行一定角度范围的随机旋转
        :param transform: 对裁剪patch做的额外变换
        """
        self.width = width
        self.height_ratio = height_ratio
        self.rotation = rotation
        self.transform = transform

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        w, h = img.size

        # 随机决定裁剪patch的宽度
        patch_w = int(round(random.uniform(self.width[0], self.width[1]) * w))
        # 随机决定裁剪patch的高度
        ratio = random.uniform(self.height_ratio[0], self.height_ratio[1])
        patch_h = int(round(ratio * patch_w))

        # 在图像中随机选起始点
        x1 = random.randint(0, max(w - patch_w, 0))
        y1 = random.randint(0, max(h - patch_h, 0))
        box = (x1, y1, x1 + patch_w, y1 + patch_h)

        # 裁剪patch
        patch = img.crop(box)

        # 随机旋转一定角度，模拟“歪斜的疤痕”
        angle = random.uniform(*self.rotation)
        patch = patch.rotate(angle, expand=True)

        if self.transform:
            patch = self.transform(patch)

        # 将旋转后的patch贴回
        augmented = img.copy()
        pw, ph = patch.size
        nx = random.randint(0, max(w - pw, 0))
        ny = random.randint(0, max(h - ph, 0))
        augmented.paste(patch, (nx, ny, nx + pw, ny + ph))

        return augmented
