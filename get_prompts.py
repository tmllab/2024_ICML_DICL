from tqdm import tqdm
import torch
import torch.nn.functional as F
import json

def get_exemplar_dict(args, exemplar_dataloader, clip_model=None, vit_g=False, classification_head=None, return_features_dict=False, return_logits_dict=False):
    exemplar_texts_dict = {}
    exemplar_images_dict = {}
    exemplar_features_dict = {}
    exemplar_logits_dict = {}
    print('Emunerating all exemplars.')
    for epoch, data in enumerate(tqdm(exemplar_dataloader)):
        batch_images, batch_texts, batch_y, batch_paths = data['images']['pixel_values'][0].to(torch.bfloat16), \
            data['texts'], data['labels'], data['image_paths']

        for each in range(batch_y.shape[0]):
            images, texts, y, batch_path = torch.unsqueeze(batch_images[each], 0), batch_texts[each], batch_y[each], batch_paths[each]
            with torch.no_grad():
                if vit_g:
                    image_embeds = clip_model.encode_image(images.to('cuda:1', torch.bfloat16))
                    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                    logits = 100. * image_embeds @ classification_head
                else:
                    image_embeds = clip_model.get_image_features(images.to('cuda:1', torch.bfloat16), return_dict=True)
                    logits = classification_head(image_embeds.to('cuda:1'))

            if y.item() not in exemplar_texts_dict.keys():
                exemplar_images_dict[y.item()] = [images.to('cuda:1')]
                if return_logits_dict:
                    exemplar_logits_dict[y.item()] = [logits.to('cuda:1')]
                if return_features_dict:
                    exemplar_features_dict[y.item()] = [image_embeds.to('cuda:1')]
            else:
                exemplar_images_dict[y.item()].append(images.to('cuda:1'))
                if return_logits_dict:
                    exemplar_logits_dict[y.item()].append(logits.to('cuda:1'))
                if return_features_dict:
                    exemplar_features_dict[y.item()].append(image_embeds.to('cuda:1'))
            exemplar_texts_dict[y.item()] = texts

    if return_features_dict:
        for y_i in exemplar_texts_dict.keys():
            exemplar_features_dict[y_i] = torch.cat(exemplar_features_dict[y_i], dim=0)
    if return_logits_dict:
        for y_i in exemplar_texts_dict.keys():
            exemplar_logits_dict[y_i] = torch.cat(exemplar_logits_dict[y_i], dim=0)
            exemplar_logits_dict[y_i] = exemplar_logits_dict[y_i].softmax(dim=1)
    
    return_list = [exemplar_images_dict, exemplar_texts_dict]
    if return_logits_dict:
        return_list.append(exemplar_logits_dict)
    if return_features_dict:
        return_list.append(exemplar_features_dict)
    return return_list


class Prompt_Builder:
    def __init__(self, args, processor, model, tran_mat, exemplar_images_dict, exemplar_texts_dict, exemplar_description_dict=None) -> None:
        image_palceholder="图"
        sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
        sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        replace_token="".join(32*[image_palceholder])
        self.exemplar_images_dict = exemplar_images_dict
        self.exemplar_texts_dict = exemplar_texts_dict
        if exemplar_description_dict:
            self.exemplar_description_dict = exemplar_description_dict
        self.tran_mat = tran_mat
 
        if args.dataset == 'imagenet':
            self.prompt = f'This image {replace_token} shows a photo of <#text>, True or False'
        elif args.dataset == 'iwildcam':
            self.prompt = f'This image {replace_token} shows a wild animal photo of <#text>, True or False'
        elif args.dataset == 'mnist':
            self.prompt = f'This image {replace_token} shows a handwritten digit photo of <#text>, True or False'
        elif args.dataset == 'domainbed' and 'Spawrious' in args.chosen_name:
            self.prompt = f'This image {replace_token} shows a dog breed photo of <#text>, True or False'
        else:
            self.prompt = f'This image {replace_token} shows a photo of <#text>, True or False'
            self.prompt_mapping = {
                'male': f'Question: Is the person in this image {replace_token} a male?',
                'wearing_hat': f'Question: Is the person in this image {replace_token} wearing a hat?',
                'smiling': f'Question: Is the person in this image {replace_token} smiling',
                'eyeglasses': f'Question: Is the person in this image {replace_token} wearing eyeglasses?',
                'blond_hair': f'Question: Does the person in this image {replace_token} have blond hair?',
                'mustache': f'Question: Does the person in this image {replace_token} have mustache?',
                'attractive': f'Question: Is the person in this image {replace_token} attractive?',
                'wearing_lipstick': f'Question: Does the person in this image {replace_token} wearing lipstick?',
                'wearing_necklace': f'Question: Does the person in this image {replace_token} wearing necklace?',
                'wearing_necktie': f'Question: Does the person in this image {replace_token} wearing necktie?',
                'young': f'Question: Is the person in this image {replace_token} young?',
                'bald': f'Question: Is the person in this image {replace_token} bald?',
                }

            self.prompt_mapping_negative = {
                'male': f'Question: Is the person in this image {replace_token} a female?',
                'wearing_hat': f'Question: Is the person in this image {replace_token} not wearing a hat?',
                'smiling': f'Question: Is the person in this image {replace_token} not smiling',
                'eyeglasses': f'Question: Is the person in this image {replace_token} not wearing eyeglasses?',
                'blond_hair': f'Question: Does the person in this image {replace_token} not have blond hair?',
                'mustache': f'Question: Does the person in this image {replace_token} not have mustache?',
                'attractive': f'Question: Is the person in this image {replace_token} not attractive?',
                'wearing_lipstick': f'Question: Does the person in this image {replace_token} not wearing lipstick?',
                'wearing_necklace': f'Question: Does the person in this image {replace_token} not wearing necklace?',
                'wearing_necktie': f'Question: Does the person in this image {replace_token} not wearing necktie?',
                'young': f'Question: Is the person in this image {replace_token} not young?',
                'bald': f'Question: Is the person in this image {replace_token} not bald?',
                }
        

            self.prompt_mapping['y'] = self.prompt_mapping[args.target_attribute.lower()]
            del self.prompt_mapping[args.target_attribute.lower()]
            self.prompt_mapping_negative['y'] = self.prompt_mapping_negative[args.target_attribute.lower()]
            del self.prompt_mapping_negative[args.target_attribute.lower()]
    
    
    def retrieve_same_logit(self, args, logits, exemplar_logits_dict, target_y):
        with torch.no_grad():
            logits_value, logits_pred = torch.max(logits.view(1, -1), dim=-1)
            logits_sim = exemplar_logits_dict[target_y].to('cuda:1')[:, logits_pred] - logits_value
        return logits_sim
    

    def retrieve_sim_logit_list(self, args, top_n_pred, logits, exemplar_logits_dict):
        retrieve_idxs = []
        logits_sims = []
        for target_y in top_n_pred:
            logits_sim = self.retrieve_same_logit(args, torch.squeeze(logits), exemplar_logits_dict, target_y)
            logits_sims.append(logits_sim.view(1, -1))
        logits_sims = torch.cat(logits_sims, dim=0)
        
        for top_n_i in range(len(top_n_pred)):
            retrieve_idxs.append([])
        retrieve_idx = torch.arange(args.num_retrieve).to(logits.device)
        for sim in range(args.num_retrieve):
            for top_n_i in range(len(top_n_pred)):
                sim_argsort = torch.argsort(logits_sims[top_n_i], dim=-1, descending=False)
                retrieve_idx = retrieve_idx[sim_argsort]
                sim_i = 0
                while retrieve_idx[sim_i] in retrieve_idxs[top_n_i]:
                    sim_i += 1
                retrieve_idxs[top_n_i].append(retrieve_idx[sim_i])

        return retrieve_idxs
    

    def retrieve_similarity(self, args, query_feature, exemplar_features_dict, target_y):
        with torch.no_grad():
            similarity = torch.cosine_similarity(query_feature, exemplar_features_dict[target_y].to('cuda:1'))
        return similarity
    

    def retrieve_sim_feature_list(self, args, top_n_pred, image_embeds, exemplar_features_dict):
        retrieve_idxs = []
        similarities = []
        for target_y in top_n_pred:
            similarity = self.retrieve_similarity(args, torch.squeeze(image_embeds), exemplar_features_dict, target_y)
            similarities.append(similarity.view(1, -1))
        similarities = torch.cat(similarities, dim=0)
        sort_sim = torch.sort(similarities, dim=-1, descending=True).values

        for top_n_i in range(len(top_n_pred)):
            retrieve_idxs.append([])
        retrieve_idx = torch.arange(args.num_retrieve).to(image_embeds.device)
        for sim in range(args.num_retrieve):
            sim_avg = torch.mean(sort_sim[:, sim])
            for top_n_i in range(len(top_n_pred)):
                sim_argsort = torch.argsort(similarities[top_n_i] - sim_avg, dim=-1, descending=False)
                retrieve_idx = retrieve_idx[sim_argsort]
                sim_i = 0
                while retrieve_idx[sim_i] in retrieve_idxs[top_n_i]:
                    sim_i += 1
                retrieve_idxs[top_n_i].append(retrieve_idx[sim_i])

        return retrieve_idxs
    

    def get_noisy_classes(self, top_n_pred, return_classes=False):
        # source noise
        self.noisy_classes = torch.argsort(self.tran_mat[:, top_n_pred], dim=0, descending=True)
        # target noise
        self.noisy_classes_hat = torch.argsort(self.tran_mat[top_n_pred, :], dim=-1, descending=True)
        if return_classes:
            return self.noisy_classes, self.noisy_classes_hat
    

    def get_inputs(self, args, images_learning, retrieve_idxs, top_n_pred, round_i=0, ex=0):
        if args.dataset != 'celebA':
            prompts, images = [], []

            if top_n_pred[round_i] == top_n_pred[0]:
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[0]]) + '? Answer: False')
                images.append(self.exemplar_images_dict[top_n_pred[(round_i+1)%len(top_n_pred)]][retrieve_idxs[(round_i+1)%len(top_n_pred)][ex]])
            else:
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[0]]) + '? Answer: False')
                images.append(self.exemplar_images_dict[top_n_pred[round_i]][retrieve_idxs[round_i][ex]])
            prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: True')
            images.append(self.exemplar_images_dict[top_n_pred[round_i]][retrieve_idxs[round_i][ex]])

            prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: ')
            images.append(images_learning)
        else:
            prompts, images = [], []
            if top_n_pred[round_i] == 0:
                prompts.append(self.prompt_mapping['y'] + '; Answer: False')
                images.append(self.exemplar_images_dict[0][ex])
                prompts.append(self.prompt_mapping_negative['y'] + '; Answer: True')
                images.append(self.exemplar_images_dict[0][ex])
            elif top_n_pred[round_i] == 1:
                prompts.append(self.prompt_mapping['y'] + '; Answer: True')
                images.append(self.exemplar_images_dict[1][ex])
                prompts.append(self.prompt_mapping_negative['y'] + '; Answer: False')
                images.append(self.exemplar_images_dict[1][ex])

            if top_n_pred[round_i] == 0:
                prompts.append(self.prompt_mapping_negative['y'] + '; Answer: ')
            elif top_n_pred[round_i] == 1:
                prompts.append(self.prompt_mapping['y'] + '; Answer: ')
            images.append(images_learning)

        images = torch.concat(images, dim=0)
        prompts = '\n'.join(prompts)
        return images, prompts


