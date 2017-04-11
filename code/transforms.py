from __future__ import division
import torch
import math
import random
from numpy import array
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import affine_transform


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)

class ToPILImage(object):
    """ Converts a torch.*Tensor of range [0, 1] and shape C x H x W
    or numpy ndarray of dtype=uint8, range[0, 255] and shape H x W x C
    to a PIL.Image of range [0, 255]
    """
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = Image.fromarray(pic)
        else:
            npimg = pic.mul(255).byte().numpy()
            npimg = np.transpose(npimg, (1,2,0))
            img = Image.fromarray(npimg)
        return img

class Normalize(object):
    """ Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""
    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number)
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        return ImageOps.expand(img, border=self.padding, fill=self.fill)

class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """
    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs):
        dimen = np.random.randint(0,2,1)
        if dimen[0] == 0:
            applied_angle  = random.uniform(0.4*self.angle, self.angle)
        else:
            applied_angle  = random.uniform(-self.angle, -0.4*self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle1_rad = angle1*np.pi/180

        inputs_np=inputs.resize_((28,28)).numpy()

        inputs_np = ndimage.interpolation.rotate(inputs_np, angle1, reshape=self.reshape, order=self.order, mode='nearest')
        inputs = torch.from_numpy(inputs_np)
        
        return inputs.resize_((1,28,28))

class RandomAffine(object):

    def __init__(self, max_angle=3.14/4, order=2):
        self.max_angle = max_angle
        self.order = order

    def __call__(self, input):

        src = input.resize_((28,28)).numpy()
        c_in = 0.5 * array(src.shape)
        dest_shape = (28, 28)
        c_out = 0.5 * array(dest_shape)

        a = random.uniform(-self.max_angle, self.max_angle)
        rot = array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        invRot = rot.T
        invScale = np.diag((1.0, 0.5))
        invTransform = np.dot(invScale, invRot)
        offset = c_in - np.dot(invTransform, c_out)
        dest = affine_transform(src, invTransform, order=self.order, offset=offset, output_shape=dest_shape, mode='nearest')

        output_tensor = torch.from_numpy(dest)

        return output_tensor.resize_((1,28,28))



class RandomSkew(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self,  max_distance=20, min_distance=8, order=3):
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.order = order

    def __call__(self, inputs):
        
        distance  = int(np.random.uniform(self.min_distance, self.max_distance))
        inputs=inputs.resize_((28,28))
        inputs_np = inputs.numpy()
        
        dimen = np.random.randint(0,2,1)
        
        h,l=28,28
        if dimen[0] == 0:
            
            def _mapping(lc):
                l,c=lc
                dec=(distance*(l-h))/h
                return l,c+dec  

            inputs_np = geometric_transform(inputs_np, mapping=_mapping, output_shape=(28, 28+distance), order=self.order, mode='nearest')
            start = (28+distance)//2 - 14
            end = (28+distance)//2 + 14
            inputs_np_center = inputs_np[:, start:end].copy()
            inputs = torch.from_numpy(inputs_np_center)
    
        
        else:
            def _mapping(lc):
                l,c=lc
                dec=(distance*(c-h))/h
                return l+dec,c  

            inputs_np = geometric_transform(inputs_np, mapping=_mapping, output_shape=(28+distance, 28), order=self.order, mode='nearest')
            start = (28+distance)//2 - 14
            end = (28+distance)//2 + 14
            inputs_np_center = inputs_np[start:end, ].copy()
            inputs = torch.from_numpy(inputs_np_center)
        
        return inputs.resize_((1,28,28))


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))
