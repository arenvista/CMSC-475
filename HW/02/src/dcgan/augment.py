from torchvision import transforms
from typing import Callable, List, Any
from PIL import Image
from enum import Enum
from functools import wraps

class TransformMode(Enum):
    SIMPLE = 1
    DELUX = 2
    def __str__(self):
        return self.name.title()

def get_transform(func: Callable[..., List[Any]]) -> Callable[..., transforms.Compose]:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> transforms.Compose:
        # func returns the list of transform operations
        transform_list = func(self, *args, **kwargs)
        # return the list wrapped in Compose
        return transforms.Compose(transform_list)
    return wrapper

class Augmentor:
    def __init__ (self, image_size: int):
        self.image_size = image_size

    @get_transform
    def simple_transform(self) -> List[Any]:
        transform_opts = [
            transforms.Resize(self.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            )
        ]
        return transform_opts

    @get_transform
    def delux_transform(self) -> List[Any]:
        transform_opts = [
            transforms.Resize(self.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
            transforms.GaussianBlur(3, 1),
        ]
        return transform_opts
