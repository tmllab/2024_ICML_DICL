import os
import sys
import torch
import argparse
import numpy as np
import templates as templates
import mydatasets as mydatasets
from transformers import CLIPModel
import myclip
from myeva.eva_clip import build_eva_model_and_transforms
from model.utils import ClassificationHead, get_zeroshot_classifier
from model.instructblip import InstructBlipProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--vit_type", type=str, default="vit-l")
parser.add_argument("--clip_ckpt", type=str, default=
    "/mnt/data3/jhuang/models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")
parser.add_argument("--processor_ckpt", type=str, default=
"/mnt/data3/jhuang/models/models--Salesforce--instructblip-flan-t5-xxl/snapshots/1a621c99c4ac000b7a4be30f78cd51040160cdc2")
parser.add_argument("--chosen_name", type=str, default='', choices=[
    'ImageNet', 'ImageNetV2', 'ImageNetA', 'ImageNetSketch', 'ImageNetR', 'ImageNetV', 'ImageNetAValClasses', 
    'ImageNetRValClasses', 'ImageNetVValClasses', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 
    "SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard",])
parser.add_argument("--dataset", type=str, default='imagenet', choices=['mnist', 'cifar10', 'iwildcam', 'celebA', 'imagenet', 'cifar100', 'domainbed'])
parser.add_argument('--targets', nargs='+', type=int, default=None, help='target domain(s) (DomainBed datasets only)')
parser.add_argument("--split", type=str, default='val')
parser.add_argument('--groupby_fields', default=['blond_hair', 'wearing_hat', 'smiling', 'eyeglasses', 'male', 'from_source_domain'])
parser.add_argument("--target_attribute", type=str, default='Eyeglasses')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--exemplar", action='store_false', default=True)
parser.add_argument("--num_exemplar", type=int, default=3)
parser.add_argument("--labeling_budget", type=int, default=50)
parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers per GPU.")
parser.add_argument("--data_location", type=str, default="/mnt/data3/datasets/mvt")
parser.add_argument("--task", type=str, default="distribution_shift")
parser.add_argument("--expname", type=str, default="")

args = parser.parse_args()
args.groupby_fields.remove(args.target_attribute.lower())
args.groupby_fields.append('y')
text_template_mapping = {
    'mnist': 'mnist_template',
    'cifar10': 'cifar_template',
    'cifar100': 'cifar_template',
    'iwildcam': 'iwildcam_template',
    'imagenet': 'openai_imagenet_template',
    'domainbed': 'cifar_template',
}

if args.vit_type == 'vit-l':
    clip_model = CLIPModel.from_pretrained(args.clip_ckpt).to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'vit-g':
    eva_clip_path = args.clip_ckpt
    model_name = "EVA_CLIP_g_14"
    clip_model, _ = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'vit-b':
    clip_model, _ = myclip.load('ViT-B/16')
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'rn-50':
    clip_model, _ = myclip.load('RN50')
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
clip_model.eval()

processor = InstructBlipProcessor.from_pretrained(
    args.processor_ckpt
)

get_image_text_loader_fn = getattr(mydatasets, 'get_' + args.dataset)
image_text_dataloader, _, n_classes, classnames = get_image_text_loader_fn(args, processor, return_classnames=True)
image_text_dataset = image_text_dataloader.dataset
text_templates = getattr(templates, text_template_mapping[args.dataset])
tran_mat = torch.zeros((n_classes, n_classes)).to('cuda:1')

classification_head = get_zeroshot_classifier(clip_model, text_templates, classnames, vit_g=args.vit_type!='vit-l').to('cuda:1', dtype=torch.bfloat16)

logits_all = []
label_all = []
clip_correct, n = 0., 0.
for epoch, data in enumerate(image_text_dataloader):
    batch_images_learning, batch_texts, batch_y = data['images']['pixel_values'][0].to(1, torch.bfloat16), data['texts'], data['labels'].to('cuda:1')
    n += batch_y.shape[0]
    with torch.no_grad():
        if args.vit_type!='vit-l':
            image_features = clip_model.encode_image(batch_images_learning.to(1, torch.bfloat16)).to('cuda:1')
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classification_head
        else:
            image_embeds = clip_model.get_image_features(batch_images_learning.to(1, torch.bfloat16), return_dict=True).to('cuda:1')
            logits = classification_head(image_embeds)
    
    label_all.append(batch_y)
    logits_all.append(logits.softmax(1))
    pred = logits.argmax(dim=1, keepdim=True)
    clip_correct += pred.eq(batch_y.view_as(pred)).sum().item()

logits_all = torch.cat(logits_all, dim=0)
label_all = torch.cat(label_all, dim=0)
confidence_all, pred_all = logits_all.max(dim=1, keepdim=True)

# Uniformly select support set based on confidence
indeces = torch.arange(len(label_all)).to('cuda:1')
for cls in range(n_classes):
    cls_idx = label_all==cls
    confidence_all_cls = confidence_all[cls_idx]
    indeces_cls = indeces[cls_idx]
    sorted_indeces_cls = indeces_cls[torch.argsort(confidence_all_cls, dim=0)]

    interval = len(sorted_indeces_cls) // args.labeling_budget

    if interval != 0:
        tran_logits = torch.mean(logits_all[sorted_indeces_cls[0:len(sorted_indeces_cls):interval]], dim=0)
    else:
        tran_logits = torch.mean(logits_all[sorted_indeces_cls], dim=0)
    tran_mat[cls] = tran_logits

max_val, min_val = tran_mat.max(), tran_mat.min()
tran_mat = (tran_mat - min_val) / (max_val - min_val)

print('CLIP Acc: ', clip_correct / n)
torch.save(tran_mat, os.path.join('tran_mat', args.expname + args.vit_type + str(args.targets[0]) + args.chosen_name + args.dataset + '_tran_mat.pt'))
print('saved tran mat')

