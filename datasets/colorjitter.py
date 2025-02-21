import numpy as np
from torchvision import transforms


class PatchColorJitter:
    """
    对图像patch执行颜色变换的示例。可以在CutPaste调用时对裁剪下来的patch单独做颜色 jitter。
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue
        )

    def __call__(self, patch):
        return self.transform(patch)
