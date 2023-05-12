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
