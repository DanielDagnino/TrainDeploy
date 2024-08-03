from typing import List
from path import Path
import torch

_EXT_IMG = ('.png', '.jpg', '.jpeg', '.bmp')
_EXT_VIDEO = ('.mp4', '.avi')  # '.mpg', '.mpeg',


def mock_image(batch_size=None, img_size=(128, 128), channels=3):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    if batch_size is None:
        return torch.randn((channels, *img_size), dtype=torch.float32)
    else:
        return torch.randn((batch_size, channels, *img_size), dtype=torch.float32)


def walk_images(path, recursive=True, ext=_EXT_IMG) -> List[Path]:
    imgs = []
    if recursive:
        for file in list(Path(path).walkfiles('*.*')):
            if file.ext.lower() in ext:
                imgs.append(file)
    else:
        for file in list(Path(path).files('*.*')):
            if file.ext.lower() in ext:
                imgs.append(file)
    return imgs


def walk_videos(path, recursive=True, ext=_EXT_VIDEO) -> List[Path]:
    return walk_images(path, recursive=recursive, ext=ext)
