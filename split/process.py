import json
import os
import shutil

input_path='/mnt/data3/datasets/ImageNet-V+'
test_path='/mnt/data3/datasets/mvt/imagenet-v-test'
examplar_path='/mnt/data3/datasets/mvt/imagenet-v-examplar'

split_file='/data/jhuang/MIC-master/split/examplar_ImageNetV_imagenet.json'
with open(split_file, 'r') as f:
    split_list=json.load(f)
split_list=[tmp.replace('/mnt/data3/jhuang/datasets/imagenet-v/', '') for tmp in split_list]

print(len(split_list))

for nfile in os.listdir(input_path):
    print(f'The {nfile} has {len(os.listdir(os.path.join(input_path, nfile)))} images.')
    for image_name in os.listdir(os.path.join(input_path, nfile)):
        source_file=os.path.join(input_path, nfile, image_name)
        if nfile+'/'+image_name in split_list:
            dest_file=os.path.join(examplar_path, nfile, image_name)
        else:
            dest_file=os.path.join(test_path, nfile, image_name)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copyfile(source_file, dest_file)