import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img_array = np.array(pic, np.int32, copy=False)
            img = torch.from_numpy(img_array)
            # backward compatibility
            img = img.float().div(self.norm_value)
            return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img_array = np.array(pic, np.int32, copy=False)
            img = torch.from_numpy(img_array)
        elif pic.mode == 'I;16':
            img_array = np.array(pic, np.int16, copy=False)
            img = torch.from_numpy(img_array)
        else:
            img_byte_storage = torch.ByteStorage.from_buffer(pic.tobytes())
            img = torch.ByteTensor(img_byte_storage)
        
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        if isinstance(img, torch.ByteTensor):
            img = img.float().div(self.norm_value)
        else:
            img = img
        
        return img

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_list = []
        for t, m, s in zip(tensor, self.mean, self.std):
            t = t.float()
            t = (t - m) / s
            tensor_list.append(t)
        return torch.stack(tensor_list)

class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            elif w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        width, height = img.size
        target_height, target_width = self.size
        
        # Calculate the starting position for the crop
        x1 = int(round((width - target_width) / 2.))
        y1 = int(round((height - target_height) / 2.))
        
        # Perform the crop using PIL's crop() function
        return img.crop((x1, y1, x1 + target_width, y1 + target_height))

class CornerCrop(object):
    def __init__(self, size, crop_position=None):
        """
        Initializes the CornerCrop transformation.

        Args:
            size (int or tuple): Desired output size of the crop. If size is an
                int, a square crop of size (size, size) will be made. If size is
                a tuple, it should contain (height, width).
            crop_position (str, optional): Specifies the crop position. Valid values
                are 'c' (center), 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left),
                'br' (bottom-right). If None, the crop position will be randomly
                selected during each transformation. Defaults to None.
        """
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        """
        Performs the CornerCrop transformation on the input image.

        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            # Center crop
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            # Top-left crop
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            # Top-right crop
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            # Bottom-left crop
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            # Bottom-right crop
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        """
        Randomly selects the crop position if randomize is True.

        This function is called to randomize the crop position during each transformation.
        """
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]
            
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()

class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to a randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to the given size.

    Args:
        scales (list or tuple): Cropping scales of the original size.
        size (int): Size of the smaller edge.
        interpolation (int, optional): Interpolation method for resizing. Default is PIL.Image.BILINEAR.
        crop_positions (list or tuple, optional): List of crop positions. Valid values are 'c' (center), 'tl' (top-left),
            'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right). Defaults to ['c', 'tl', 'tr', 'bl', 'br'].
    """

    def __init__(self, scales, size, interpolation=Image.BILINEAR, crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation
        self.crop_positions = crop_positions

    def __call__(self, img):
        # Calculate the minimum length of the image size
        min_length = min(img.size[0], img.size[1])

        # Randomly select a scale from the available scales
        self.scale = random.choice(self.scales)

        # Calculate the crop size based on the scale
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        # Randomly select a crop position from the available crop positions
        self.crop_position = random.choice(self.crop_positions)

        if self.crop_position == 'c':
            # Center crop
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            # Top-left crop
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            # Top-right crop
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            # Bottom-left crop
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            # Bottom-right crop
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        # Perform the crop operation
        img = img.crop((x1, y1, x2, y2))

        # Resize the cropped image to the desired size
        img = img.resize((self.size, self.size), self.interpolation)

        return img
    
    def randomize_parameters(self):
        """
        Randomly selects the scale and crop position during each transformation.
        """
        self.scale = random.choice(self.scales)
        self.crop_position = random.choice(self.crop_positions)
