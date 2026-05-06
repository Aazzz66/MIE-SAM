import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

#from skimage import exposure


def cv_random_flip(img: object, t, gt):
    flip_flag = random.randint(0, 1)
    # flip_flag1= random.randint(0, 1)
    # flip_flag2= random.randint(0, 1)
    
    #left right flip
    if flip_flag == 1:
    # if flip_flag1 == 1:
        img    = img.transpose(Image.FLIP_LEFT_RIGHT)
        t      = t.transpose(Image.FLIP_LEFT_RIGHT)
        gt     = gt.transpose(Image.FLIP_LEFT_RIGHT)

    
    # #top bottom flip
    # if flip_flag2==1:
    #     img    = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     t      = t.transpose(Image.FLIP_TOP_BOTTOM)
    #     gt     = gt.transpose(Image.FLIP_TOP_BOTTOM)
    #     body   = body.transpose(Image.FLIP_TOP_BOTTOM)
    #     detail = detail.transpose(Image.FLIP_TOP_BOTTOM)
    return img,t,gt
def randomCrop(img,t,gt):
    border=30
    image_width = img.size[0]
    image_height = img.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return img.crop(random_region), t.crop(random_region),gt.crop(random_region)
def randomRotation(img,t,gt):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        img    = img.rotate(random_angle, mode)
        t      = t.rotate(random_angle, mode)
        gt     = gt.rotate(random_angle, mode)

    return img,t,gt
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)  
        randY=random.randint(0,img.shape[1]-1)  
        if random.randint(0,1)==0:  
            img[randX,randY]=0  
        else:  
            img[randX,randY]=255 
    return Image.fromarray(img)  


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, train_root, trainsize):
        self.trainsize = trainsize
       #self.max_samples = max_samples  # 最大样本数

        self.image_root  = train_root + '/RGB/'
        self.gt_root     = train_root + '/GT/'
        self.t_root      = train_root + '/T/'

        

        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts    = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png')]
        self.ts     = [self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')]


        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.ts     = sorted(self.ts)

        #self.images = sorted(self.images)[:max_samples]
        #self.gts = sorted(self.gts)[:max_samples]
        #self.ts = sorted(self.ts)[:max_samples]

        self.augmentation = albu.Compose([

            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),

            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(trainsize, trainsize),
        ], additional_targets={
            't': 'image',
            'gt': 'mask'
        })
        # self.filter_files()
        self.size = len(self.images)

        ## RGB(VT5000 + VT1000 + VT821)
        # [0.525, 0.590, 0.537], [0.177, 0.167, 0.176]

        ## MIX(VT5000 + VT1000 + VT821)
        # [0.501, 0.612, 0.602], [0.173, 0.152, 0.166]


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])])


        ## T(VT5000 + VT1000 + VT821)
        # [0.736, 0.346, 0.339], [0.179, 0.196, 0.169]

        self.t_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])])

        self.gt_transform     = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        t = self.rgb_loader(self.ts[index])
        gt = self.binary_loader(self.gts[index])

        # 转换为numpy数组进行增强
        image_np = np.array(image)
        t_np = np.array(t)
        gt_np = np.array(gt)

        # 应用相同的增强
        augmented = self.augmentation(
            image=image_np,
            t=t_np,
            mask=gt_np
        )

        image_aug = Image.fromarray(augmented['image'])
        t_aug = Image.fromarray(augmented['t'])
        gt_aug = Image.fromarray(augmented['mask'])

        image = self.img_transform(image_aug)
        t = self.t_transform(t_aug)
        gt = self.gt_transform(gt_aug).squeeze()

        return image, t, gt


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def __len__(self):
        return self.size

#dataloader for training
def get_loader(train_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset(train_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset(data.Dataset):
    def __init__(self, test_root, testsize):
        self.testsize = testsize

        self.image_root = test_root + '/RGB/'
        self.gt_root    = test_root + '/GT/'
        self.t_root     = test_root + '/T/'

        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts    = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.ts     = [self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.ts     = sorted(self.ts)

        ## RGB(VT5000 + VT1000 + VT821)
        # [0.525, 0.590, 0.537], [0.177, 0.167, 0.176]
        ## MIX(VT5000 + VT1000 + VT821)
        # [0.501, 0.612, 0.602], [0.173, 0.152, 0.166]

        ## RGB(VT1606)
        # [0.238, 0.271, 0.236], [0.172, 0.174, 0.174]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])])
        

        ## T(VT5000 + VT1000 + VT821)
        # [0.736, 0.346, 0.339], [0.179, 0.196, 0.169]
        ## T(VT251)
        # [0.273, 0.687, 0.716], [0.148, 0.212, 0.155]

        ## T(VT1606)
        # color
        # [0.213, 0.656, 0.779], [0.113, 0.203, 0.122]
        # gray
        # [0.3451, 0.345, 0.345], [0.0768, 0.0768, 0.0768]


        self.t_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])])

        #self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),  # 确保GT尺寸与输入一致
            transforms.ToTensor()
        ])

        self.size = len(self.images)
        #self.index = 0


    def __getitem__(self,index):
        image = self.rgb_loader(self.images[index])
        shape = image.size
        image = self.img_transform(image)#.unsqueeze(0)

        t = self.rgb_loader(self.ts[index])
        t = self.t_transform(t)#.unsqueeze(0)

        gt = self.binary_loader(self.gts[index])
        gt = self.gt_transform(gt)#.unsqueeze(0)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        #self.index += 1
        #self.index = self.index % self.size
        return image, t, gt, shape, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size