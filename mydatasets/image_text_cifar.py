from typing import Any, Callable, Optional, Tuple
import torchvision.datasets as tv_dataset
from PIL import Image
import numpy as np

class image_text_CIFAR(tv_dataset.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = True, exemplar=False, num_exemplar=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.label_to_class_mapping = {
            0 : "airplane",
            1 : "automobile",
            2 : "bird",
            3 : "cat",
            4 : "deer",
            5 : "dog",
            6 : "frog",
            7 : "horse",
            8 : "ship",
            9 : "truck",
        }

        self.num_classes = max(self.targets)+1
        self.targets = np.array(self.targets)
        self.data = np.array(self.data)
        self.exemplar = exemplar

        if exemplar:
            exemplar_indeces = []
            total_indeces = np.arange(len(self.data))
            for i in range(self.num_classes):
                cls_idx = total_indeces[self.targets == i]
                select_idx = np.random.choice(cls_idx, size=num_exemplar, replace=False)
                exemplar_indeces.append(select_idx)
            exemplar_indeces = np.concatenate(exemplar_indeces, axis=0)
            self.indices = np.array(exemplar_indeces)

            self.data = self.data[self.indices]
            self.targets = self.targets[self.indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(images=img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        text = self.label_to_class_mapping[target]

        return {
            "images": img,
            "texts": text,
            'index': str(index),
            "labels": target,
            "image_paths": str(index)
        }


    def __len__(self) -> int:
        return len(self.data)
    

class image_text_CIFAR100(tv_dataset.CIFAR100):
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = True, exemplar=False, num_exemplar=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.fine_label_names = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "cra",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
        self.label_to_class_mapping = {}
        for i, name in enumerate(self.fine_label_names):
            self.label_to_class_mapping[i] = name

        self.num_classes = max(self.targets)+1
        self.targets = np.array(self.targets)
        self.data = np.array(self.data)
        self.exemplar = exemplar

        if exemplar:
            exemplar_indeces = []
            total_indeces = np.arange(len(self.data))
            for i in range(self.num_classes):
                cls_idx = total_indeces[self.targets == i]
                select_idx = np.random.choice(cls_idx, size=num_exemplar, replace=False)
                exemplar_indeces.append(select_idx)
            exemplar_indeces = np.concatenate(exemplar_indeces, axis=0)
            self.indices = np.array(exemplar_indeces)

            self.data = self.data[self.indices]
            self.targets = self.targets[self.indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(images=img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        text = self.label_to_class_mapping[target]

        return {
            "images": img,
            "texts": text,
            'index': str(index),
            "labels": target,
            "image_paths": str(index)
        }


    def __len__(self) -> int:
        return len(self.data)
    


class image_text_MNIST(tv_dataset.MNIST):
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = True, exemplar=False, num_exemplar=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.label_to_class_mapping = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }

        self.num_classes = max(self.targets)+1
        self.targets = np.array(self.targets)
        self.data = np.array(self.data)
        self.exemplar = exemplar

        if exemplar:
            exemplar_indeces = []
            total_indeces = np.arange(len(self.data))
            for i in range(self.num_classes):
                cls_idx = total_indeces[self.targets == i]
                select_idx = np.random.choice(cls_idx, size=num_exemplar, replace=False)
                exemplar_indeces.append(select_idx)
            exemplar_indeces = np.concatenate(exemplar_indeces, axis=0)
            self.indices = np.array(exemplar_indeces)

            self.data = self.data[self.indices]
            self.targets = self.targets[self.indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(images=img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        text = self.label_to_class_mapping[target]

        return {
            "images": img,
            "texts": text,
            'index': str(index),
            "labels": target,
            "image_paths": str(index)
        }


    def __len__(self) -> int:
        return len(self.data)