from PIL import Image
import os
# from imagenetv2_pytorch import ImageNetV2Dataset

from .imagenet import ImageNet

# class ImageNetV2DatasetWithPaths(ImageNetV2Dataset):
#     def __getitem__(self, i):
#         img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
#         if self.transform is not None:
#             img = self.transform(img)
#         return {
#             'images': img,
#             'labels': label,
#             'texts': self.classnames[label],
#             'image_paths': str(self.fnames[i])
#         }

class ImageNetV2(ImageNet):
    # def get_test_dataset(self):
    #     return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location, exemplar=self.exemplar, num_exemplar=self.num_exemplar, dataset_type=self.dataset_type)    
    def get_test_path(self):
        test_path = os.path.join(self.location, 'imagenet-v2-'+self.dataset_type)
        return test_path
