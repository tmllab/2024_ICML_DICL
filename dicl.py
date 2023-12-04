import os
import sys
import json
import time
import torch
import myclip
import argparse
import mydatasets
import vlm_forward
import templates as templates
from transformers import CLIPModel
from myeva.eva_clip import build_eva_model_and_transforms
from model.utils import ClassificationHead, get_zeroshot_classifier
from accelerate.utils import get_balanced_memory
from accelerate import init_empty_weights, infer_auto_device_map
from get_prompts import get_exemplar_dict, Prompt_Builder
from model.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor

sys.path.append('/data/jhuang/Otter-main/')
sys.path.append("/data/jhuang/Otter-main/src/otter_ai/models")
from otter.modeling_otter import OtterForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="mmicl", choices=['mmicl', 'otter'])
parser.add_argument("--vit_type", type=str, default="vit-l", choices=['vit-l', 'vit-g', 'vit-b', 'rn-50'])
parser.add_argument("--model_ckpt", type=str, default=
    "/mnt/data3/jhuang/models/models--BleachNick--MMICL-Instructblip-T5-xxl/snapshots/ed4ddb6c60ff260c3c03ff149b7e91ce3496690e")
parser.add_argument("--clip_ckpt", type=str, default=
    "/mnt/data3/jhuang/models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")
parser.add_argument("--processor_ckpt", type=str, default=
    "/mnt/data3/jhuang/models/models--Salesforce--instructblip-flan-t5-xxl/snapshots/1a621c99c4ac000b7a4be30f78cd51040160cdc2")
parser.add_argument("--chosen_name", type=str, default='', choices=[
    'ImageNet', 'ImageNetV2', 'ImageNetA', 'ImageNetSketch', 'ImageNetR', 'ImageNetV', 'ImageNetAValClasses', 
    'ImageNetRValClasses', 'ImageNetVValClasses', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 
    "SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard",])
parser.add_argument("--dataset", type=str, default='imagenet', choices=['mnist', 'cifar10', 'iwildcam', 'celebA', 'imagenet', 'cifar100', 'domainbed'])
parser.add_argument('--targets', nargs='+', type=int, default=[0], help='target domain(s) (DomainBed datasets only)')
parser.add_argument("--split", type=str, default='val')
parser.add_argument('--groupby_fields', default=[
    'male', 'wearing_hat', 'smiling', 'eyeglasses', 'blond_hair', 'mustache', 'attractive', 
    'wearing_lipstick','wearing_necklace', 'wearing_necktie', 'young', 'bald', 'from_source_domain'])
parser.add_argument("--target_attribute", type=str, default='Eyeglasses')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--top_n", type=int, default=6)
parser.add_argument("--exemplar", action='store_false', default=True)
parser.add_argument("--num_exemplar", type=int, default=10)
parser.add_argument("--num_retrieve", type=int, default=3)
parser.add_argument("--task", type=str, default="distribution_shift")
parser.add_argument("--stop_iteration", type=int, default=5000)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers per GPU.")
parser.add_argument("--data_location", type=str, default="/mnt/data3/datasets/mvt")
parser.add_argument("--expname", type=str, default="")
args = parser.parse_args()
config = InstructBlipConfig.from_pretrained(args.model_ckpt)
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

if 'mmicl' in args.model_type:
    with init_empty_weights():
        model = InstructBlipForConditionalGeneration(config)
        max_memory = get_balanced_memory(
            model,
            max_memory={0: '23000MB', 1: '23000MB', 2: '23000MB', 3: '23000MB'},
            dtype=None,
            low_zero=False,
        )
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["T5Block"], dtype=torch.bfloat16)
    

    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_ckpt,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        config=config,)
elif 'otter' in args.model_type:
    model = OtterForConditionalGeneration.from_pretrained("/mnt/data3/liuchang/otter_ckpt")
    max_memory = get_balanced_memory(
        model,
        max_memory={0: '10000MB', 1: '20000MB'},
        dtype=None,
        low_zero=False,
    )
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["T5Block"], dtype=torch.bfloat16)
    model = OtterForConditionalGeneration.from_pretrained(
        "/mnt/data3/liuchang/otter_ckpt", 
        device_map=device_map,)
    model=model.to(torch.bfloat16)
    model.text_tokenizer.padding_side = "left"

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

processor = InstructBlipProcessor.from_pretrained(
    args.processor_ckpt
)

get_image_text_loader_fn = getattr(mydatasets, 'get_' + args.dataset)
if args.dataset != 'celebA':
    image_text_dataloader, exemplar_dataloader, n_classes, classnames = get_image_text_loader_fn(args, processor, return_classnames=True)
    assert n_classes == len(classnames)
    image_text_dataset = image_text_dataloader.dataset

    text_templates = getattr(templates, text_template_mapping[args.dataset])
    classification_head = get_zeroshot_classifier(clip_model, text_templates, classnames, vit_g=args.vit_type!='vit-l').to('cuda:1', dtype=torch.bfloat16)

    tran_mat = torch.load(os.path.join('tran_mat', args.expname + args.vit_type + str(args.targets[0]) + args.chosen_name + args.dataset + '_tran_mat.pt'), map_location='cuda:1')
else:
    image_text_dataloader, n_classes = get_image_text_loader_fn(args, processor)
    tran_mat = None

exemplar_images_dict, exemplar_texts_dict, exemplar_logits_dict, exemplar_features_dict = get_exemplar_dict(args, exemplar_dataloader, clip_model=clip_model, vit_g=args.vit_type!='vit-l', classification_head=classification_head, return_features_dict=True, return_logits_dict=True)
prompt_builder = Prompt_Builder(args, processor, model, tran_mat, exemplar_images_dict, exemplar_texts_dict)
vlm_forward_fn = getattr(vlm_forward, 'vlm_forward_' + args.model_type)

model.eval()
clip_model.eval()

logits_finetune=[]
mvt_correct, clip_correct, n = 0., 0., 0.
for epoch, data in enumerate(image_text_dataloader):
    batch_images_learning, batch_texts, batch_y, input_paths = data['images']['pixel_values'][0].to('cuda:1', torch.bfloat16), data['texts'], data['labels'], data['image_paths']

    for each in range(batch_y.shape[0]):
        input_path=input_paths[each]
        n += 1
        images_learning, texts, y = torch.unsqueeze(batch_images_learning[each], 0), batch_texts[each], batch_y[each]
        with torch.no_grad():
            if args.vit_type!='vit-l':
                image_features = clip_model.encode_image(images_learning.to(1, torch.bfloat16)).to('cuda:1')
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ classification_head
            else:
                image_embeds = clip_model.get_image_features(images_learning.to(1, torch.bfloat16), return_dict=True).to('cuda:1')
                logits = classification_head(image_embeds)

        logits = logits[0,:].softmax(0)
        max_score = torch.max(logits).item()
        clip_pred = logits.argsort(dim=-1, descending=True).tolist()

        ########## Using Image and Prompt to do ICL ###########
        top_n_pred, _ = prompt_builder.get_noisy_classes(clip_pred[0], True)
        top_n_sort = torch.argsort(logits[top_n_pred], dim=-1, descending=True)
        top_n_pred = top_n_pred[top_n_sort].tolist()
        if clip_pred[0] not in top_n_pred:
            top_n_pred.insert(0, clip_pred[0])
        top_n_pred = top_n_pred[:args.top_n]

        top_n_list = []
        for p in top_n_pred:
            if p in exemplar_texts_dict.keys():
                top_n_list.append(p)
        top_n_pred = top_n_list[:args.top_n]

        retrieve_idxs = prompt_builder.retrieve_sim_logit_list(args, top_n_pred, logits, exemplar_logits_dict)

        compare_gen = []
        compare_logit = []
        compare_exemplar = []
        compare_pred = []
        for i in range(len(top_n_pred)):
            logit_dict = {'true': 0., 'fal': 0.}
            for ex in range(args.num_retrieve):
                images, prompts = prompt_builder.get_inputs(args, images_learning, retrieve_idxs, top_n_pred, round_i=i, ex=ex)
                logit = vlm_forward_fn(model, images, prompts, processor)
                sorted_logit = torch.argsort(logit, dim=-1, descending=True)

                # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
                if args.model_type == 'mmicl':
                    generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
                    generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
                    softmax_value = torch.Tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
                elif args.model_type == 'otter':
                    generated_text='true'
                    generated_text2='fal'
                    softmax_value = torch.Tensor([logit[:, 5852], logit[:, 7700]]).softmax(0)

                logit_dict[generated_text.lower()] += (softmax_value[0].item() / args.num_retrieve)
                logit_dict[generated_text2.lower()] += (softmax_value[1].item() / args.num_retrieve)

            logit_value = torch.Tensor([logit_dict['true'], logit_dict['fal']])
            if logit_value[0] > 0.5:
                generated_text = 'true'
                generated_text2 = 'fal'
            else:
                generated_text = 'fal'
                generated_text2 = 'true'

            if args.dataset == 'cifar10':
                if i == 0 and (logit_value[0].item() + max_score) / 2 >= 0.6:
                    vlm_text = exemplar_texts_dict[clip_pred[0]]
                    compare_logit.append(logit_value[0].item())
                    compare_logit = torch.tensor(compare_logit)
                    break
            else:
                if i == 0 and (logit_value[0].item() + max_score) / 2 >= args.threshold:
                    vlm_text = exemplar_texts_dict[clip_pred[0]]
                    compare_logit.append(logit_value[0].item())
                    compare_logit = torch.tensor(compare_logit)
                    break

            compare_gen.append(generated_text.lower())
            compare_logit.append(logit_value[0].item())
            compare_exemplar.append(exemplar_texts_dict[top_n_pred[i]])
            compare_pred.append(top_n_pred[i])

            print('output: {}, \t\tlogit: {:.3f}-{:.3f}, \t\tgt: {}, \t\tpred: {}'.format(
                generated_text, logit_value[0].item(), logit_value[1].item(), 
                texts, exemplar_texts_dict[top_n_pred[i]]))

        if len(compare_gen) > 0:
            compare_logit = torch.tensor(compare_logit)
            pred_sort_idx = torch.argsort(compare_logit, dim=0, descending=True)
            vlm_text = compare_exemplar[pred_sort_idx[0]]
            vlm_pred = compare_pred[pred_sort_idx[0]]

        if vlm_text == texts:
            mvt_correct += 1
        if clip_pred[0] == y:
            clip_correct += 1
            
        print('iter:', n, '; pred:', vlm_text, '; vlm:', mvt_correct, '; clip:', clip_correct)
        logits_finetune.append({input_path: [compare_pred, compare_logit.cpu().tolist()]})

    if n >= args.stop_iteration:
        break

timestamp = time.time()
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
print('MVT Acc: ', mvt_correct / n)
print('CLIP Acc: ', clip_correct / n)

## save logits for fine-tune
output_dict={'mvt_acc': mvt_correct / n, 'clip_acc': clip_correct / n, 'logits': logits_finetune}
with open(f'logits/{args.expname}_logits2finetune_{args.vit_type}_{str(args.targets[0])}_{args.chosen_name}_{args.dataset}_{time_str}.json', 'w') as f:
    json.dump(output_dict, f)
    json.dump(vars(args), f)
