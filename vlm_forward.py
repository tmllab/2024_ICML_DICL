import torch

def vlm_forward_mmicl(model, images, prompts, processor):
    inputs = processor(images=None, text=prompts, max_length=1024, padding='max_length', return_tensors="pt")

    inputs['pixel_values'] = images
    inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            pixel_values = inputs['pixel_values'],
            input_ids = inputs['input_ids'].to('cuda:3'),
            decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device='cuda:3') * 0,
            attention_mask = inputs['attention_mask'].to('cuda:3'),
            img_mask = inputs['img_mask'],
            return_dict = True,
            output_attentions = False,
            output_hidden_states = False,
        )
    logit = outputs.language_model_outputs.logits[:, -1, :]
    
    return logit


def get_formatted_prompt(prompt: str, in_context_prompts: list = []) -> str:
    in_context_string = ""
    for in_context_prompt, in_context_answer in in_context_prompts:
        in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
    return f"{in_context_string}<image>User: {prompt} GPT:<answer>"

def vlm_forward_otter(model, images, prompts, processor):
    images=images.unsqueeze(1).unsqueeze(0)
    in_context_prompts = []
    cls_name=prompts[prompts.find('photo of')+9: prompts.find(', True or False')]
    in_context_prompts.append((f'This image shows a photo of {cls_name}, True or False?', 'False'))
    in_context_prompts.append((f'This image shows a photo of {cls_name}, True or False?', 'True'))

    prompts_input = f'This image shows a photo of {cls_name}, True or False?'

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompts_input, in_context_prompts),
        ],
        return_tensors="pt",
    )
    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    outputs = model.forward(
        vision_x=images.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
    )
    logits=outputs['logits']
    return logits[:,-1,:]
