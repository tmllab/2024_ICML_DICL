

from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig
)
import torch
from tqdm import tqdm
import clip.clip as clip
from model.blip2.modeling_blip_2 import Blip2ForConditionalGeneration
from model.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration,
    "instructblip": InstructBlipForConditionalGeneration,

}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )


    for param in model.parameters():
        param.requires_grad = False

    for param in model.language_projection.parameters():
        param.requires_grad = True

    if model_args.backbone_model == 'flan-t5':
        for block in model.language_model.encoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True

        for block in model.language_model.decoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True
            block.layer[1].EncDecAttention.q.requires_grad=True
            block.layer[1].EncDecAttention.v.requires_grad=True
    else:# vicuna
        print(f"vicuna layer:{len(model.language_model.model.layers)}")
        for block in model.language_model.model.layers:
            block.self_attn.q_proj.weight.requires_grad=True
            block.self_attn.v_proj.weight.requires_grad=True
    
    all_param = 0
    trained_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad ==True:
            trained_param+=param.numel()
    total_param = all_param 

    print('***** total param is {} *****'.format(total_param))
    print('***** total trained param is {} *****'.format(trained_param))
    return model




class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)
    

def get_zeroshot_classifier(clip_model, templates, classnames=None, tokenizer=None, processor=None, vit_g=False):
    if not vit_g:
        logit_scale = clip_model.logit_scale

        with torch.no_grad():
            zeroshot_weights = []
            print('Enumerating all classes.')
            for classname in tqdm(classnames):
                texts = []
                for t in templates:
                    texts.append(t(classname.replace("_", " ")))
                texts = clip.tokenize(texts).to('cuda:1')  # tokenize
                # embeddings = clip_model.encode_text(texts)
                embeddings = clip_model.get_text_features(texts)  # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()
                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to('cuda:1')
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()
            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(normalize=True,
                                                weights=zeroshot_weights)
        return classification_head
    
    else:
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for t in templates:
                    texts.append(t(classname))
                texts = clip.tokenize(texts).to('cuda:1')  # tokenize
                class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to('cuda:1')
        return zeroshot_weights