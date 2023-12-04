import os
import sys
import json
import time
import copy
import torch
import myclip
import pickle
import argparse
import mydatasets
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
from torch import optim
import templates as templates
import torch.backends.cudnn as cudnn
from timm.utils import accuracy, AverageMeter
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from mydatasets.utils.ina import imagenet_a_mask
from mydatasets.utils.inr import imagenet_r_mask
from mydatasets.utils.inv import imagenet_v_mask
from myeva.eva_clip import build_eva_model_and_transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from .utils import AverageMeter, create_logger, create_scheduler

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

class MVTDataset(Dataset):
    def __init__(self, args, meta_file, preprocessor):
        self.args=args
        self.meta_file = meta_file
        self.preprocessor = preprocessor
        self.results=[]
        for line in open(meta_file, 'r'):
            line=line.strip()
            self.results.append(line.split('; '))
    def __getitem__(self, index):
        info=self.results[index]
        image_dir=info[0]
        if self.args.dataset=='imagenet-c':
            image_dir=image_dir.replace('ImageNet-test-5k/val', f'imagenet-c-test-5k/{self.args.corruption}/{self.args.severity}')
        image = Image.open(image_dir).convert("RGB")
        image = self.preprocessor(image)
        label = int(info[1])

        return image, label, image_dir

    def __len__(self):
        return len(self.results)

class MVTCifarDataset(Dataset):
    def __init__(self, meta_file, preprocessor):
        self.data=[]
        self.targets = []
        with open(meta_file, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            self.data.append(entry["data"])
            if "labels" in entry:
                self.targets.extend(entry["labels"])
            else:
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.preprocessor = preprocessor
    def __getitem__(self, index):
        images, targets = self.data[index], self.targets[index]
        images = Image.fromarray(images)
        images = self.preprocessor(images)

        return images, targets, str(index)

    def __len__(self) -> int:
        return len(self.data)

class MVTMnistDataset(MNIST):
    def __init__(self, root, train = False, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, str(index)
    
def zeroshot_classifier(clip_model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = myclip.tokenize(texts).cuda()  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main(args):
    setup_seeds(0)
    args.output_dir=os.path.join(args.output_dir, f'{args.dataset}_{args.chosen_name}_{str(args.targets[0])}_train1')
    os.makedirs(args.output_dir, exist_ok=True)
    logger=create_logger(output_dir=args.output_dir, dist_rank=0, name='Training qformer.')

    # settings
    settings={
        'imagenet-val': {'meta_file': './mydatasets/data_paths/ImageNet-test-5k.txt', 'mask': None, 'teacher_json': 'logits2finetune_ImageNet_imagenet_2023-11-13 20:06:15.json'},
        'imagenet-v2': {'meta_file': './mydatasets/data_paths/imagenet-v2-test-5k.txt', 'mask': None, 'teacher_json': 'logits2finetune_ImageNetV2_imagenet_2023-11-14 00:26:45.json'},
        'imagenet-a': {'meta_file': './mydatasets/data_paths/imagenet-a-test-5k.txt', 'mask': imagenet_a_mask, 'teacher_json': 'logits2finetune_vit-l_0_ImageNetA_imagenet_2023-11-18 02:40:17.json'},
        'imagenet-r': {'meta_file': './mydatasets/data_paths/imagenet-r-test-5k.txt', 'mask': imagenet_r_mask, 'teacher_json': 'logits2finetune_vit-l_0_ImageNetR_imagenet_2023-11-18 03:07:01.json'},
        'imagenet-sketch': {'meta_file': './mydatasets/data_paths/imagenet-sketch-test-5k.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_0_ImageNetSketch_imagenet_2023-11-18 03:37:57.json'},
        'imagenet-v': {'meta_file': './mydatasets/data_paths/imagenet-v-test-5k.txt', 'mask': imagenet_v_mask, 'teacher_json': 'logits2finetune_vit-l_0_ImageNetV_imagenet_2023-11-18 04:04:39.json'},
        'cifar10': {'meta_file': '/mnt/data3/datasets/cifar10/cifar-10-batches-py/test_batch', 'mask': None, 'teacher_json': 'logits2finetune_ImageNetV2_cifar10_2023-11-14 20:30:44.json'},
        'cifar100': {'meta_file': '/mnt/data3/datasets/cifar100/cifar-100-python/test', 'mask': None, 'teacher_json': 'logits2finetune_ImageNetV_cifar100_2023-11-14 06:57:53.json'},
        'mnist': {'meta_file': '/mnt/data3/datasets/mnist/MNIST', 'mask': None, 'teacher_json': 'logits2finetune__mnist_2023-11-15 07:33:34.json'},
        'wilds': {'meta_file': './mydatasets/data_paths/vitl-wilds-1w.txt', 'mask': None, 'teacher_json': 'logits2finetune_ImageNet_iwildcam_2023-11-11 16:02:46.json'},
        'domainbed': {
            'VLCS': [{'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[0]_VLCS_domainbed_2023-11-15 18:01:28.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[1]_VLCS_domainbed_2023-11-15 17:39:28.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[2]_VLCS_domainbed_2023-11-15 17:45:06.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[3]_VLCS_domainbed_2023-11-15 18:12:09.json'}
                ],
            'PACS': [{'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[0]_PACS_domainbed_2023-11-15 20:08:05.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[1]_PACS_domainbed_2023-11-15 20:58:33.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[2]_PACS_domainbed_2023-11-15 21:48:28.json'},
                     {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[3]_PACS_domainbed_2023-11-15 22:15:18.json'}
                ],
            'OfficeHome': [{'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[0]_OfficeHome_domainbed_2023-11-15 23:11:02.json'},
                           {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[1]_OfficeHome_domainbed_2023-11-16 00:43:50.json'},
                           {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[2]_OfficeHome_domainbed_2023-11-16 02:34:55.json'},
                           {'meta_file': None, 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[3]_OfficeHome_domainbed_2023-11-16 04:25:17.json'}
                ],
            'DomainNet': [{'meta_file': './mydatasets/data_paths/vitl-domainnet0.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[0]_DomainNet_domainbed_2023-11-16 00:18:48.json'},
                          {'meta_file': './mydatasets/data_paths/vitl-domainnet1.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[1]_DomainNet_domainbed_2023-11-16 03:23:07.json'},
                          {'meta_file': './mydatasets/data_paths/vitl-domainnet2.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[2]_DomainNet_domainbed_2023-11-16 00:13:15.json'},
                          {'meta_file': './mydatasets/data_paths/vitl-domainnet3.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[3]_DomainNet_domainbed_2023-11-16 02:23:46.json'},
                          {'meta_file': './mydatasets/data_paths/vitl-domainnet4.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_[4]_DomainNet_domainbed_2023-11-16 13:24:59.json'}
                ],
        },
        'imagenet-c': {'meta_file': './mydatasets/data_paths/ImageNet-test-5k.txt', 'mask': None, 'teacher_json': 'logits2finetune_vit-l_0_ImageNetC_imagenet'}
    }

    if args.dataset=='domainbed':
        meta_file=settings[args.dataset][args.chosen_name][args.targets[0]]['meta_file']
        mask=settings[args.dataset][args.chosen_name][args.targets[0]]['mask']
        teacher_json=settings[args.dataset][args.chosen_name][args.targets[0]]['teacher_json']
    else:
        meta_file=settings[args.dataset]['meta_file']
        mask=settings[args.dataset]['mask']
        teacher_json=settings[args.dataset]['teacher_json']

    # create clip_model and preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.vit_type == 'vit-l':
        clip_model, preprocess = myclip.load('ViT-L/14')
        clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
    elif args.vit_type == 'vit-g':
        eva_clip_path = args.clip_ckpt
        model_name = "EVA_CLIP_g_14"
        clip_model, preprocess = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)
        clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
    elif args.vit_type == 'vit-b':
        clip_model, preprocess = myclip.load('ViT-B/16')
        clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
    elif args.vit_type == 'rn-50':
        clip_model, preprocess = myclip.load('RN50')
        clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)


    # create dataset
    if args.dataset=='domainbed':
        dataset = MVTDataset(meta_file, preprocess)
        train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        if args.dataset=='cifar10' or args.dataset=='cifar100':
            dataset = MVTCifarDataset(meta_file, preprocess)
        elif args.dataset=='mnist':
            dataset = MVTMnistDataset(root=meta_file, transform=preprocess)
        else:
            dataset = MVTDataset(args, meta_file, preprocess)
        train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    text_template_mapping = {
        'mnist': 'mnist_template',
        'cifar10': 'cifar_template',
        'cifar100': 'cifar_template',
        'iwildcam': 'iwildcam_template',
        'imagenet': 'openai_imagenet_template',
        'domainbed': 'cifar_template',
    }
    templates = getattr(templates, text_template_mapping[args.dataset])
    get_image_text_loader_fn = getattr(mydatasets, 'get_' + args.dataset)
    _, _, _, class_names = get_image_text_loader_fn(args, preprocess, return_classnames=True)

    zeroshot_weights = zeroshot_classifier(clip_model, class_names, templates)
    zeroshot_weights=zeroshot_weights.to(torch.float32)
    clip_model=clip_model.visual
    clip_model=clip_model.to(torch.float32)
    clip_model.eval()
    
    # create teacher clip_model and logits
    teacher_model=copy.deepcopy(clip_model)
    teacher_model.eval()
    for k, v in teacher_model.named_parameters():
        v.requires_grad = False

    with open(os.path.join('logits/', teacher_json), 'r') as f:
        logits_json=json.load(f)

    # Format of teacher_logits:
    # image_paths: [[topk_image_labels], [topk_image_logits]]
    teacher_logits={}
    for tmp in logits_json['logits']:
        key_dir=list(tmp.keys())[0]
        teacher_logits[key_dir]=tmp[key_dir]

    # optimizer
    optim_kwargs={
        # 'lr': 1e-6,
        'lr': 5e-7,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 5e-7,
    }
    optimizer = optim.Adam(params= filter(lambda p: p.requires_grad, clip_model.parameters()), **optim_kwargs)
    
    # scheduler
    logger.info('Initialize scheduler.')
    updates_per_epoch = len(train_dataloader)
    sched_kwargs={
        'sched': 'cosine',
        # 'num_epochs': 10,
        'num_epochs': 3,
        'warmup_epochs': 0,
        'min_lr': 5e-8,
        'step_on_epochs': False,
        'updates_per_epoch': updates_per_epoch
    }
    scheduler, num_epochs = create_scheduler(optimizer, **sched_kwargs)

    # evaluate first
    val_acc = evaluation_epoch(args, -1, num_epochs, val_dataloader, mask, zeroshot_weights, clip_model, logger)
    logger.info(f'First evaluation, acc is {val_acc}.')

    # training epochs
    for epoch in range(num_epochs):
        # clip_model training
        train_loss, batch_time = train_epoch(args, epoch, num_epochs, train_dataloader, mask, zeroshot_weights, clip_model, teacher_model, teacher_logits, optimizer, scheduler, logger)
        # clip_model evaluation
        val_acc = evaluation_epoch(args, epoch, num_epochs, val_dataloader, mask, zeroshot_weights, clip_model, logger)

        # record loss
        logger.info(f'Finish training epoch {epoch}. The train loss is {train_loss}, the val acc is {val_acc}, the batch time is {batch_time}.')

        # save state dict
        checkpoint={'clip_model': clip_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'lr': scheduler.state_dict()}
        torch.save(checkpoint, os.path.join(args.output_dir, f'save{epoch}.pth'))
        
    logger.info('Finish training.')


def get_logits(images, zeroshot_weights, teacher_model):
    with torch.no_grad():
        image_features = teacher_model(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ zeroshot_weights

    return logits


def train_epoch(args, epoch, num_epochs, train_dataloader, mask, zeroshot_weights, clip_model, teacher_model, teacher_logits, optimizer, scheduler, logger):
    # settings
    num_updates=epoch*len(train_dataloader)
    train_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    timer = time.time()
    crossentropy=torch.nn.CrossEntropyLoss()
    log_scale=torch.ones([]) * np.log(1 / 0.07)
    log_scale.cuda()
    
    # to train mode
    clip_model.train()
    for batch_id, data in enumerate(train_dataloader):
        images, targets, image_paths = data[0], data[1], data[2]
        images = images.cuda()
        targets = targets.cuda()
        batch_size = images.shape[0]

        # forward and backward
        image_features = clip_model(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        new_logits = image_features @ zeroshot_weights
        new_logits=log_scale.exp()*new_logits
        if mask is not None:
            new_logits = new_logits[:, mask]

        # obtain teacher logits
        original_logits = get_logits(images, zeroshot_weights, teacher_model)
        original_logits=log_scale.exp()*original_logits
        if mask is not None:
            # mask out the reduced classes in some ImageNet variant datasets.
            original_logits = original_logits[:, mask].detach()
        else:
            original_logits = original_logits.detach()        
        
        # obtain loss for label equivalent
        losses=torch.zeros((batch_size,), dtype=torch.float).cuda()
        for sample_id, (new_logit, original_logit) in enumerate(zip(new_logits, original_logits)):
            new_logit=new_logit.unsqueeze(0)
            image_path=image_paths[sample_id]
            if image_path in teacher_logits.keys():
                modified_logits=teacher_logits[image_path]
            else:
                modified_logits=None
            if modified_logits is None or len(modified_logits[0])==0:
                _, top1_label=torch.topk(original_logit, 1)
                losses[sample_id]=crossentropy(new_logit, top1_label)
            else:
                topk_label=torch.tensor(modified_logits[0], dtype=torch.long).cuda()
                topk_logit=torch.tensor(modified_logits[1], dtype=torch.float).cuda()
                _, top1_id=torch.topk(topk_logit, 1)
                top1_label=topk_label[top1_id.item()]
                top1_label=top1_label.unsqueeze(0)
                losses[sample_id]=crossentropy(new_logit, top1_label)

        loss=torch.mean(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_m.update(loss.item(), batch_size)
        num_updates += 1
        scheduler.step_update(num_updates=num_updates, metric=train_loss_m.avg)
        batch_time_m.update(time.time() - timer)
        timer = time.time()
        
        # log loss and batch time
        if (batch_id+1) % args.log_interval ==0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            logger.info(f'Training epoch {epoch+1}/{num_epochs}, batch {batch_id}/{len(train_dataloader)}, the train loss is {train_loss_m.avg}, the batch time is {batch_time_m.avg}, the learning rate is {lr}.')
    
    return train_loss_m.avg, batch_time_m.avg

@torch.no_grad()
def evaluation_epoch(args, epoch, num_epochs, val_dataloader, mask, zeroshot_weights, clip_model, logger):
    # settings
    val_acc_m = AverageMeter()
    softmax = torch.nn.Softmax(dim=1)
    clip_model.eval()

    # evaluation
    for batch_id, data in enumerate(val_dataloader):
        images, targets, image_paths = data[0], data[1], data[2]
        images = images.cuda()
        targets = targets.cuda()
        batch_size = images.shape[0]

        # predict
        image_features = clip_model(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ zeroshot_weights
        logits = softmax(logits)
        if mask is None:
            acc1, _ = accuracy(logits, targets, topk=(1, 5))
        else:
            acc1, _ = accuracy(logits[:,mask], targets, topk=(1, 5))

        acc1/=100
        val_acc_m.update(acc1.item(), images.size(0))

        # log loss
        if (batch_id+1) % args.log_interval ==0:
            logger.info(f'Evaluating epoch {epoch+1}/{num_epochs}, batch {batch_id}/{len(val_dataloader)}, the val acc is {val_acc_m.avg}.')

    return val_acc_m.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_type", type=str, default="vit-l", choices=['vit-l', 'vit-g', 'vit-b', 'rn-50'])
    parser.add_argument("--dataset", type=str, default='imagenet', choices=['mnist', 'cifar10', 'iwildcam', 'celebA', 'imagenet', 'cifar100', 'domainbed'])
    parser.add_argument("--chosen_name", type=str, default='', choices=[
        'ImageNet', 'ImageNetV2', 'ImageNetA', 'ImageNetSketch', 'ImageNetR', 'ImageNetV', 'ImageNetAValClasses', 
        'ImageNetRValClasses', 'ImageNetVValClasses', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 
        "SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard",])
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers per GPU.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--data_location', default='/mnt/data3/datasets/mvt', type=str)
    parser.add_argument('--targets', nargs='+', type=int, default=[0], help='target domain(s) (DomainBed datasets only)')
    parser.add_argument("--num_exemplar", type=int, default=10)
    parser.add_argument('--corruption', default='brightness', type=str)
    parser.add_argument('--severity', default='0', type=str)

    args = parser.parse_args()
    main(args)
