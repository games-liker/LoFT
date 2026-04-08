import os
from .lt_data import LT_Dataset


class ImageNet_LT(LT_Dataset):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    train_txt = "./datasets/ImageNet_LT/ImageNet_LT_train.txt"
    test_txt = "./datasets/ImageNet_LT/ImageNet_LT_test.txt"
    # test_txt = "./datasets/ImageNet_LT/imagenet-sketch.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()
        self.img_path = []
        self.labels = []

        self.names = []
        with open(self.txt) as f:
            for line in f:
                file_name = line.split()[0]
                self.img_path.append(os.path.join(root, ("train" if train else "test") + "_imgnet", file_name.split('/')[2]))
                self.labels.append(int(line.split()[1]))
                self.names.append(self.classnames[int(line.split()[1])])
        self.cls_num_list = self.get_cls_num_list()
        

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames