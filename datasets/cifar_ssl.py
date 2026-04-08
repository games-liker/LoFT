import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

from collections import defaultdict

import torch.utils.data as data
import os
import sys
import pickle

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
resolution = 224

def get_cifar10(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base_dataset = datasets.CIFAR10(root, train=True, download=True)


    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=mean, std=std, size=resolution))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    l_samples = make_imb_data(args.num_max, 100, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, 100, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, 100)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=mean, std=std, size=resolution))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_smallimagenet(args, root):
    assert args.img_size == 32 or args.img_size == 64, 'img size should only be 32 or 64!!!'
    base_dataset = SmallImageNet(root, args.img_size, True)

    labeled_percent = 0.01 # 设置为1%

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # select labeled data and construct labeled dataset
    num_classes = len(set(base_dataset.targets))
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in base_dataset.targets:
        num_data_per_cls[l] += 1

    num_labeled_data_per_cls = [int(np.around(n * labeled_percent)) for n in num_data_per_cls]
    print('total number of labeled data is ', sum(num_labeled_data_per_cls))

    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_data_per_cls, args, num_classes)

    train_labeled_dataset = SmallImageNet(root, args.img_size, True, transform=transform_train, indexs=train_labeled_idxs)
    train_unlabeled_dataset = SmallImageNet(root, args.img_size, True,
                                            transform=TransformFixMatch(mean=mean, std=std, size=resolution))
    test_dataset = SmallImageNet(root, args.img_size, False, transform=transform_val)

    arr = np.array(num_labeled_data_per_cls)
    tar_index = np.argsort(-arr)
    tar_index = tar_index.tolist()

    for idx in range(len(train_labeled_dataset.targets)):
        train_labeled_dataset.targets[idx] = tar_index.index(train_labeled_dataset.targets[idx])

    for idx in range(len(train_unlabeled_dataset.targets)):
        train_unlabeled_dataset.targets[idx] = tar_index.index(train_unlabeled_dataset.targets[idx])

    for idx in range(len(test_dataset.targets)):
        test_dataset.targets[idx] = tar_index.index(test_dataset.targets[idx])

    train_unlabeled_dataset.targets = np.array(train_unlabeled_dataset.targets)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_coco(args, root):
    # 从args/cfg中读取COCO文件列表路径,如果没有则使用默认值
    coco_file_list = getattr(args, 'coco_file_list', None)
    if coco_file_list is None:
        # 默认路径,使用相对路径
        default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'COCO_imagenet_good_HC0.6.txt')
        if os.path.exists(default_path):
            coco_file_list = default_path
        else:
            # 如果默认路径不存在,尝试使用绝对路径(向后兼容)
            coco_file_list = "datasets/COCO_imagenet_good_HC0.6.txt"
            if not os.path.exists(coco_file_list):
                raise FileNotFoundError(f"COCO file list not found. Please specify 'coco_file_list' in config or place the file at: {default_path}")
    
    train_unlabeled_dataset = COCO(root, transform=TransformFixMatch(mean=mean, std=std, size=resolution), coco_file_list=coco_file_list)
    return train_unlabeled_dataset

from .places_lt import Places_LT
def get_place(args, root):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(resolution * 8 // 7),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_labeled_dataset = Places_LT(root, train=True, transform=transform_train)
    # 从args/cfg中读取Places相关路径配置
    places_unlabeled_file = getattr(args, 'places_unlabeled_file', None)
    places_root = getattr(args, 'places_root', None)
    if places_root is None:
        places_root = root  # 默认使用root作为places_root
    train_unlabeled_dataset = Place_Unlabeled(root, transform=TransformFixMatch(mean=mean, std=std, size=resolution),
                                               places_unlabeled_file=places_unlabeled_file, places_root=places_root)
    test_dataset = Places_LT(root, train=False, transform=transform_test)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs

def train_split_l(labels, n_labeled_per_class, args, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
    return train_labeled_idxs

def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    print(class_num_list)
    return list(class_num_list)

def x_u_split(args, labels, num_classes):
    label_per_class = args.num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class Test(object):
    def __init__(self, mean, std, size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), strong

class TransformFixMatch(object):
    def __init__(self, mean, std, size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformLOFT(object):
    def __init__(self, mean, std, size=224):
        self.weak = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
            ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        
        self.classnames = self.classes
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.targets:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
    
class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, transform=None, target_transform=None, indexs=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1

        self.targets = [i - 1 for i in self.targets]
        self.data = np.vstack(self.data).reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC  shape(-1, 32, 32, 3)


        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        
        self.num_classes = len(set(self.targets))
        self.cls_num_list = self.get_cls_num_list()
        self.classnames = []
        # 尝试从多个可能的路径读取类别名称文件
        classnames_file = getattr(self, 'classnames_file', None)
        if classnames_file is None:
            # 尝试相对路径
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'construct_small_imagenet_127', 'synset_words_up_down_127.txt')
            if os.path.exists(default_path):
                classnames_file = default_path
            else:
                # 向后兼容的绝对路径
                classnames_file = 'construct_small_imagenet_127/synset_words_up_down_127.txt'
        
        if os.path.exists(classnames_file):
            with open(classnames_file) as f:
                for line in f:
                    self.classnames.append(' '.join(line.split()[1:]))
        else:
            # 如果文件不存在,使用数字作为类别名称
            self.classnames = [f'class_{i}' for i in range(self.num_classes)]
            print(f"Warning: Class names file not found at {classnames_file}, using default class names")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index
        return img, target


    def __len__(self):
        return len(self.data)
    

    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.targets:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list

class Place_Unlabeled(data.Dataset):
    def __init__(self, file_path, transform=None, places_unlabeled_file=None, places_root=None):

        self.place_unlabeled_path = []
        # 尝试从多个可能的路径读取Places unlabeled文件列表
        if places_unlabeled_file is None:
            # 尝试相对路径
            default_path = os.path.join(os.path.dirname(__file__), 'Places_LT', 'Places_unlabeled.txt')
            if os.path.exists(default_path):
                places_unlabeled_file = default_path
            else:
                # 向后兼容的路径
                places_unlabeled_file = "datasets/Places_LT/Places_unlabeled.txt"
        
        if os.path.exists(places_unlabeled_file):
            with open(places_unlabeled_file) as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        self.place_unlabeled_path.append(line.split()[0])
        else:
            raise FileNotFoundError(f"Places unlabeled file not found at: {places_unlabeled_file}")

        self.transform = transform
        self.root = file_path
        # Places数据集根目录,如果没有指定则使用默认值
        self.places_root = places_root if places_root is not None else ""
        

    def __len__(self):
        return len(self.place_unlabeled_path)

    def __getitem__(self, index):

        path = os.path.join(self.places_root, self.place_unlabeled_path[index])

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # return image
        return image, -1


class COCO(data.Dataset):
    def __init__(self, file_path, transform=None, coco_file_list=None):
        """
        Args:
            file_path: COCO图像文件的根目录(目前未使用,保留用于向后兼容)
            transform: 图像变换
            coco_file_list: COCO文件列表的路径,包含要使用的图像文件路径列表
        """
        self.img_filenames = os.listdir(file_path) if os.path.exists(file_path) else []

        # 从参数中读取COCO文件列表路径
        if coco_file_list is None:
            # 默认路径,使用相对路径
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'COCO_imagenet_good_HC0.6.txt')
            if os.path.exists(default_path):
                coco_file_list = default_path
            else:
                # 如果默认路径不存在,尝试使用绝对路径(向后兼容)
                coco_file_list = "datasets/COCO_imagenet_good_HC0.6.txt"
        
        # 读取COCO文件列表
        self.coco_good_path = []
        if os.path.exists(coco_file_list):
            with open(coco_file_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        self.coco_good_path.append(line.split()[0])
        else:
            raise FileNotFoundError(f"COCO file list not found at: {coco_file_list}")

        self.transform = transform
        self.root = file_path
        

    def __len__(self):
        return len(self.coco_good_path)

    def __getitem__(self, index):

        path = self.coco_good_path[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, -1

# class 

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'smallimagenet': get_smallimagenet,
                   'coco': get_coco,
                   'place': get_place}
