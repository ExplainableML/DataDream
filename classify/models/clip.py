"""
Code mainly from https://github.com/vturrisi/disef/blob/main/fine-tune/src/model.py
"""

import types
import torch
import torch.nn as nn
import clip
from timm.models._manipulate import checkpoint_seq

from models.lora import lora_replace_attention_layers
from util_data import SUBSET_NAMES, TEMPLATES_SMALL

def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet_100": "",
        "imagenet": "",
        "std10": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
    }[dataset]
    return dataset_name


class CLIP(nn.Module):
    def __init__(
        self, 
        dataset,
        is_lora_image,
        is_lora_text,
        clip_download_dir="model_clip",
        clip_version="ViT-B/16",
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)
        self.is_lora_image = is_lora_image
        self.is_lora_text = is_lora_text
        self.clip_version = clip_version

        # TODO: change the number of templates
        self.templates = TEMPLATES_SMALL[:1]

        self.clip, _ = clip.load(clip_version, device="cpu", download_root=clip_download_dir)

        # visual model
        if is_lora_image:
            if self.clip_version != "RN50":
                self.clip.visual.transformer = lora_replace_attention_layers(
                    self.clip.visual.transformer,
                    lora_r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    start_block=0,
                )

        # text model
        if is_lora_text:
            self.clip.transformer = lora_replace_attention_layers(
                self.clip.transformer,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                start_block=0,
            )

        self.register_buffer("tokenized_text", self.tokenize_text())

        # enable checkpointing for text transformer
        # datasets with more classes simply go OOM if we don't do this
        def checkpoint_forward(self, x):
            x.requires_grad = True
            x = checkpoint_seq(self.resblocks, x)
            return x

        self.clip.transformer.forward = types.MethodType(
            checkpoint_forward, self.clip.transformer)

        # configure all learnable parameters
        self.set_learnable_params()

#     @staticmethod
    def tokenize_text(self):
        print("Tokenizing text...")

        texts = []
        for classname in SUBSET_NAMES[self.dataset]:

            class_texts = []
            for template in self.templates:
                class_texts.append(
                    template.format(self.dataset_name, classname))

            class_texts = clip.tokenize(class_texts)

            texts.append(class_texts)

        texts = torch.stack(texts)
        return texts

    def set_learnable_params(self):
        # turn off all parameters
        for p in self.clip.parameters():
            p.requires_grad = False

        # learnable parameters for the visual model
        if self.is_lora_image:
            if self.clip_version != "RN50":
                for name, p in self.clip.visual.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True
            else:
                for name, p in self.clip.visual.named_parameters():
                    p.requires_grad = True
                
#         elif not self.cfg.freeze_visual:
#             for p in self.clip.visual.parameters():
#                 p.requires_grad = True

        # learnable parameters for the text model
        if self.is_lora_text:
            for name, p in self.clip.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    def learnable_params(self):
#         return [{"name": "all", "params": [p for p in self.clip.parameters() if p.requires_grad]}]
        return [p for p in self.clip.parameters() if p.requires_grad]

    def forward_image(
        self,
        x: torch.Tensor,
    ):
        image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text):
        n_classes, n_prompts, n_token = tokenized_text.size()
#         tokenized_text = einops.rearrange(tokenized_text, "c p d -> (c p) d")
        tokenized_text = tokenized_text.view(-1, n_token)
        with torch.set_grad_enabled(self.is_lora_text):
            text_feats = self.clip.encode_text(tokenized_text)

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # average across multiple prompt templates and re-norm
#         text_feats = einops.rearrange(text_feats, "(c p) d -> c p d", c=n_classes, p=n_prompts)
        text_feats = text_feats.view(n_classes, n_prompts, -1)
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return text_feats

    def forward(
        self,
        x: torch.Tensor,
        tokenized_text: torch.Tensor = None,
        output_features: bool = False,
        **kwargs,
    ):
        if tokenized_text is None:
            tokenized_text = self.tokenized_text

        image_feats = self.forward_image(x)
        text_feats = self.forward_text(tokenized_text)

        logit_scale = self.clip.logit_scale.exp()

        # no instance-specific text feats
        if len(text_feats.shape) == 2:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_feats @ text_feats.t()
        else:
            logits_per_image = logit_scale * torch.stack(
                [image_feats[i] @ text_feats[i].t() for i in range(image_feats.shape[0])]
            )

        if output_features:
            return {
                "logits": logits_per_image,
                "image_feats": image_feats,
                "text_feats": text_feats,
            }

        return logits_per_image

