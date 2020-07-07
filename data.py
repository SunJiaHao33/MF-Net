import os
import cv2
import torch
import numpy as np
import scipy.misc as misc
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

#随机色相饱和度值
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

#随机移位刻度旋转
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

#随机水平翻转
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

#随机垂直翻转
def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

#随机90旋转
def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

#随机放大
def random_zoom(image, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        image = image.apply_affine_transform(image, zx=zx, zy=zy)
        mask = image.apply_affine_transform(mask, zx=zx, zy=zy)
    return image, mask

#随机裁剪
def random_shear(image, mask, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        image = image.apply_affine_transform(image, shear=sh)
        mask = image.apply_affine_transform(mask, shear=sh)
    return image, mask

#
def random_gray(image, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(image * coef, axis=2)
        image = np.dstack((gray, gray, gray))
    return image

#随机对比度
def random_contrast(image, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha * image + gray
        image = np.clip(image, 0., 1.)
    return image

#随机亮度
def random_brightness(image, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        image = alpha * image
        image = np.clip(image, 0., 1.)
    return image


#随机通道变化
def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_augmentation(img, mask):
    
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    
    
    img, mask = randomHorizontalFlip(img, mask)                                   
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)                              

    return img, mask









def default_DRIVE_loader(img_path, mask_path, image_size=(448, 448), mode='train', transform=None, target_transform=None):
    img = cv2.imread(img_path)
    mask = np.array(Image.open(mask_path))

    # resize
    img = cv2.resize(img, image_size)
    mask = cv2.resize(mask, image_size)

    if mode == 'train':
        img, mask = random_augmentation(img, mask)
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

    elif mode == 'test':
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask



def default_CHASEDB1_loader(img_path, mask_path, image_size=(960, 960), mode='train'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)

    mask = cv2.imread(mask_path)
    # print(mask_path, mask.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mode == 'train':
        mask = cv2.resize(mask, image_size) # 注意训练的时候才reszie mask
        img, mask = random_augmentation(img, mask)
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

    elif mode == 'test':
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1)/255*3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

    return img, mask

def default_STARE_loader(img_path, mask_path, image_size=(576, 576), mode='train'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    mask = np.array(Image.open(mask_path))

    if mode == 'train':
        mask = cv2.resize(mask, image_size) # 注意训练的时候才reszie mask
        img, mask = random_augmentation(img, mask)
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

    elif mode == 'test':
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1)/255*3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

    return img, mask

def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        image_root = os.path.join(root_path, 'training/images')
        gt_root = os.path.join(root_path, 'training/1st_manual')

    if mode == 'test':
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/1st_manual')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_CHASEDB1_datasets(root_path, mode='train'):

    images = []
    masks = []

    if mode == 'train':
        image_root = os.path.join(root_path, 'training/images')
        gt_root = os.path.join(root_path, 'training/1st_manual')

    elif mode =='test':
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/1st_manual')

    for image_name in os.listdir(image_root):

        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '_1stHO.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_STARE_datasets(root_path, mode = 'train'):

    images = []
    masks = []

    if mode == 'train':
        image_root = os.path.join(root_path, 'training/images')
        gt_root = os.path.join(root_path, 'training/labels')

    elif mode =='test':
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/labels')

    for image_name in os.listdir(image_root):

        image_path = os.path.join(image_root, image_name.split('.')[0] + '.ppm')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.ah.ppm')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


class ImageFolder(data.Dataset):
    def __init__(self, datasets='DRIVE', image_size=(448, 448), mode='train', transform=None, target_transform=None):
        self.mode = mode
        self.image_size = image_size
        self.dataset = datasets
        self.transform = transform
        self.target_transform = target_transform

        assert self.dataset in ['DRIVE', 'CHASEDB1', 'STARE'], \
            "the dataset should be in 'DRIVE', CHASEDB1' "

        if self.dataset == 'DRIVE':
            self.images, self.labels = read_DRIVE_datasets('./dataset/DRIVE', self.mode)

        elif self.dataset == 'CHASEDB1':
            self.images, self.labels = read_CHASEDB1_datasets('./dataset/CHASEDB1', self.mode)

        elif self.dataset == 'STARE':
            self.images, self.labels = read_STARE_datasets('./dataset/STARE', self.mode)
        else:
            print('Default dataset is DRIVE')
            self.images, self.labels = read_DRIVE_datasets('./dataset/DRIVE', self.mode)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])



        if self.mode == 'train':
            image, label = random_augmentation(np.asarray(image), np.asarray(label))

            image = Image.fromarray(image)
            label = Image.fromarray(label)

        # if self.mode == 'test':
        #     image = np.asarray(image)
        #     label = np.asarray(label)
        #     H, W, C = image.shape
        #     top = int((H - 544) / 2)
        #     bottom = 544 + top
        #     left = int((W - 544) / 2)
        #     right = 544 + left
        #     image = Image.fromarray(image[top:bottom, left:right, :])
        #     label = Image.fromarray(label[top:bottom, left:right])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

