import os
import torch

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
import numpy as np
import torchvision
import numpy as np
from PIL import Image
import glob


class CustomDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transforms = transform
        self.class_list = sorted(os.listdir(root))
        self.img_list = []
        self.class_len_list = []
        for i, c in enumerate(self.class_list):
            root_child = os.path.join(root, c)
            self.img_list.append(sorted(glob.glob(root_child + "/*")))
            self.class_len_list.append(len(self.img_list[-1]))

    def __len__(self):
        total_len = 0
        for i, c in enumerate(self.class_list):
            total_len += len(self.img_list[i])
        return total_len

    def __getitem__(self, idx):
        batch_img = []
        # batch_y = []
        for i, c in enumerate(self.class_list):
            rand_idx = np.random.randint(0, self.class_len_list[i])
            img_name = self.img_list[i][rand_idx]
            image = self.transforms(Image.open(img_name))
            batch_img.append(image)
            # batch_y.append(i)

        batch_img = torch.stack(batch_img, dim=0)

        return batch_img


class ImageNetC:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser('~/data'),
        batch_size=32,
        num_workers=32,
        classnames='openai',
        custom=False,
        exemplar=False, 
        num_exemplar=None,
        dataset_type=None,
        corruption='brightness',
        severity='1',
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.n_classes = len(self.classnames)
        self.custom = custom
        self.exemplar = exemplar
        self.num_exemplar = num_exemplar
        self.dataset_type = dataset_type
        self.corruption=corruption
        self.severity=severity
        # self.populate_train()
        self.populate_test()

    def set_corruption(self, corruption, severity):
        self.corruption=corruption
        self.severity=severity
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, 'ImageNet-'+self.dataset_type, 'train')
        self.train_dataset = ImageFolderWithPaths(traindir,
                                                  transform=self.preprocess, 
                                                  classnames=self.classnames,
                                                  exemplar=self.exemplar, 
                                                  num_exemplar=self.num_exemplar)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

        if self.custom:
            self.train_dataset_custom = CustomDataset(
                root=traindir, transform=self.preprocess,)
            self.train_loader_custom = torch.utils.data.DataLoader(
                self.train_dataset_custom,
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers)

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler())

    def get_test_path(self):
        # test_path = os.path.join(self.location, 'ILSVRC2015', 'train_val_split_val')
        test_path = os.path.join(self.location, 'imagenet-c-'+self.dataset_type, self.corruption, self.severity)
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, 'imagenet-c-'+self.dataset_type, self.corruption, self.severity)
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(),
                                    transform=self.preprocess,
                                    classnames=self.classnames,
                                    exemplar=self.exemplar,
                                    num_exemplar=self.num_exemplar)

    def name(self):
        return 'imagenet-c'

