from PIL import Image
from typing import Tuple, Optional, Sequence, Union

import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

from tllib.vision.transforms import ResizeImage

DEFAULT_SCALE = [0.08, 1.0]
DEFAULT_RATIO = [3. / 4., 4. / 3.]

def _get_resizing_transform(mode: str = 'default', pre_crop_size: Union[int, Sequence] = 256, 
                            resize_size: Union[int, Sequence] = 224, 
                            scale: Optional[Sequence[float]] = (0.08, 1.0), 
                            ratio: Optional[Sequence[float]] = (3. / 4., 4. / 3.)) -> Tuple[T.Compose, int]:
    if scale is None:
        scale = DEFAULT_SCALE
    if ratio is None:
        scale = DEFAULT_RATIO
        
    if mode == 'default':
        transform = T.Compose([
            ResizeImage(pre_crop_size),
            T.RandomResizedCrop(resize_size, scale=scale, ratio=ratio)
        ])
        transformed_img_size = resize_size
    elif mode == 'cen.crop':
        transform = T.Compose([
            ResizeImage(pre_crop_size),
            T.CenterCrop(resize_size)
        ])
        transformed_img_size = resize_size
    elif mode == 'ran.crop':
        transform = T.Compose([
            ResizeImage(pre_crop_size),
            T.RandomCrop(resize_size)
        ])
        transformed_img_size = resize_size
    elif mode == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(mode)
    return transform, transformed_img_size


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transform, transformed_img_size = _get_resizing_transform(mode=resizing, resize_size=resize_size, scale=scale, ratio=ratio)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)) -> T.Compose:
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        - res.: resize the image to 224
    """
    transform, _ = _get_resizing_transform(mode=resizing, resize_size=resize_size)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
