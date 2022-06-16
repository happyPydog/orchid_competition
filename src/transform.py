import random
from torchvision import transforms as T
from PIL import ImageFilter, ImageOps

INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)
ORCHID_MEAN = (0.4909, 0.4216, 0.3703)
ORCHID_STD = (0.2459, 0.2420, 0.2489)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = T.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def train_tf(
    img_size: int = 224, three_data_aug: bool = False, color_jitter: float = None
) -> T.Compose:

    primary_tf1 = [
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
    ]

    if three_data_aug:
        primary_tf1 += [
            T.RandomChoice(
                [gray_scale(p=0.5), Solarization(p=0.5), GaussianBlur(p=0.5)]
            )
        ]

    if color_jitter:
        primary_tf1.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))

    final_tfl = [T.ToTensor(), T.Normalize(mean=ORCHID_MEAN, std=ORCHID_STD)]

    return T.Compose(primary_tf1 + final_tfl)


def test_tf(img_size: int = 224) -> T.Compose:
    size = int((256 / 224) * img_size)
    return T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=ORCHID_MEAN, std=ORCHID_STD),
        ]
    )
