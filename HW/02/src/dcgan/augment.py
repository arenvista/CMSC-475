from torchvision import transforms
from PIL import Image
from enum import Enum
from functools import wraps

class TransformMode(Enum):
    SIMPLE = 1
    DELUX = 2
    def __str__(self):
        return self.name.title()

def get_transform(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        result = transforms.Compose(result)
        return result
    return wrapper

class Augmentor:
    def __init__ (self, image_size: int):
        self.image_size = image_size

    @get_transform
    def simple_transform(self):
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
    def delux_transform(self):
        transform_opts = []
        return transform_opts
