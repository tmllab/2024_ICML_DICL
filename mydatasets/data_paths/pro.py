import os
import json

# input_mvt_dir='/mnt/data3/datasets/mvt/'
# output_mvt_dir='/data/liuchang/mmbench/datasets/mvt'

# dataset_names=['ImageNet-test-5k/val', 'imagenet-v2-test-5k', 'imagenet-a-test-5k', 'imagenet-r-test-5k', 'imagenet-sketch-test-5k', 'imagenet-v-test-5k']

# for dataset_name in dataset_names:
#     if dataset_name=='ImageNet-test-5k/val':
#         output_dataset_name='ImageNet-test-5k.txt'
#     else:
#         output_dataset_name=dataset_name+'.txt'
#     nfile_list=os.listdir(os.path.join(input_mvt_dir, dataset_name))
#     nfile_list.sort()
#     with open(os.path.join(output_mvt_dir, output_dataset_name), 'w') as f:
#         for i, nfile in enumerate(nfile_list):
#             for image_name in os.listdir(os.path.join(input_mvt_dir, dataset_name, nfile)):
#                 image_path=os.path.join(input_mvt_dir, dataset_name, nfile, image_name)
#                 f.write(image_path+'; '+str(i)+'\n')

input_mvt_dir='/mnt/data3/datasets/mvt/domain_net'
output_mvt_dir='/data/liuchang/mmbench/datasets/mvt'
logits_dir='/data/liuchang/mmbench/jhuang-mic/logits_rv/'

domainnet_all= {'rn50-domainnet0': '_logits2finetune_rn-50_0_DomainNet_domainbed_2023-11-26 11:58:41.json',
                'rn50-domainnet1': '_logits2finetune_rn-50_1_DomainNet_domainbed_2023-11-26 14:46:42.json',
                'rn50-domainnet2': '_logits2finetune_rn-50_2_DomainNet_domainbed_2023-11-26 15:46:03.json',
                'rn50-domainnet3': '_logits2finetune_rn-50_3_DomainNet_domainbed_2023-11-26 15:14:13.json',
                'rn50-domainnet4': '_logits2finetune_rn-50_4_DomainNet_domainbed_2023-11-26 17:57:47.json'
                }

folders=os.listdir(os.path.join(input_mvt_dir, 'clipart'))
folders.sort()

for domainnet_name in domainnet_all.keys():
    with open(os.path.join(output_mvt_dir, domainnet_name+'.txt'), 'w') as f:
        with open(os.path.join(logits_dir, domainnet_all[domainnet_name]), 'r') as logits_f:
            logits_json=json.load(logits_f)
        for tmp in logits_json['logits']:
            key_dir=list(tmp.keys())[0]
            label=folders.index(key_dir.split('/')[-2])
            f.write(key_dir+'; '+str(label)+'\n')
