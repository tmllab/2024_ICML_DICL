import os
import torch
import torchvision.datasets as datasets

from .imagenet import ImageNetSubsample, ImageNetSubsampleValClasses
import numpy as np


CLASS_SUBLIST = [403, 404, 412, 413, 422, 425, 427, 428, 430, 437, 456, 465, 471, 479, 481, 483, 484, 487, 492, 497, 504, 505, 508, 515, 517, 519, 521, 526, 528, 535, 547, 555, 559, 561, 567, 569, 571, 579, 589, 603, 606, 609, 621, 628, 640, 650, 652, 657, 658, 670, 671, 672, 673, 681, 682, 695, 698, 703, 716, 721, 736, 742, 745, 750, 751, 755, 759, 761, 763, 765, 770, 779, 782, 807, 817, 831, 843, 846, 847, 849, 859, 861, 873, 879, 882, 883, 895, 915, 919, 920, 928, 930, 933, 934, 945, 948, 951, 959, 963, 966]
CLASS_SUBLIST_MASK = [(i in CLASS_SUBLIST) for i in range(1000)]


class ImageNetVValClasses(ImageNetSubsampleValClasses):
    def get_class_sublist_and_mask(self):
        return CLASS_SUBLIST, CLASS_SUBLIST_MASK

class ImageNetV(ImageNetSubsample):
    def get_class_sublist_and_mask(self):
        return CLASS_SUBLIST, CLASS_SUBLIST_MASK

    def get_test_path(self):
        return os.path.join(self.location, 'imagenet-v-'+self.dataset_type)
    