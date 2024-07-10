
import json
import os
import pickle
import random
import string
from os.path import join as ospj
from typing import Tuple

import fire
import numpy as np
import torch
import torchvision as tv
import yaml
from safetensors import safe_open
from tqdm import tqdm

from util import (
    batch_iteration, 
    make_dirs, 
    set_seed,
    SUBSET_NAMES,
    TEMPLATES_SMALL,
)

def get_pipe(model_type, model_dir, device, is_tqdm):
    # CUDA_VISIBLE_DEVICES issue
    # https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
    from diffusers import DiffusionPipeline, StableDiffusionPipeline

    if model_type not in ("sdxl1.0",):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=not is_tqdm)

    return pipe


def get_prompt_embeds(pipe, prompts, device):
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if (
        hasattr(pipe.text_encoder.config, "use_attention_mask")
        and pipe.text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def update_pipe(
    pipe,
    n_shot,
    n_template,
    dataset,
    datadream_dir,
    datadream_lr,
    datadream_epoch,
    datadream_train_text_encoder,
    fewshot_seed,
    classname,
):

    if datadream_dir is None:
        raise ValueError("`datadream_dir` should be defined.")

    print("Update pipe with DataDream.")
    mid = f"shot{n_shot}_{fewshot_seed}_tpl{n_template}"
    if not datadream_train_text_encoder:
        mid += "_notextlora"
    fpath = ospj(
        datadream_dir,
        dataset,
        mid,
        f"lr{datadream_lr}_epoch{datadream_epoch}",
        classname,
    )
    pipe.load_lora_weights(fpath, weight_name="pytorch_lora_weights.safetensors")


    return pipe


def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet": "",
        "imagenet_100": "",
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


@torch.no_grad()
def get_text_embeds_for_weight(pipe, device, dataset, prompt2="both"):
    dataset_name = get_dataset_name_for_template(dataset)
    embeds_original = []
    embeds_soft = []
    for template in TEMPLATES_SMALL:
        prompts_original = [
            template.format(dataset_name, clsname) for clsname in SUBSET_NAMES[dataset]
        ]

        if prompt2 == "both":
            prompts_soft = [
                template.format(dataset_name, f"<{clsname}>, {clsname}")
                for clsname in SUBSET_NAMES[dataset]
            ]
        elif prompt2 == "short":
            prompts_soft = [
                template.format(dataset_name, f"<{clsname}>")
                for clsname in SUBSET_NAMES[dataset]
            ]
        else:
            raise ValueError('`prompt2` should be either "both" or "short".')

        _embeds_original = get_prompt_embeds(pipe, prompts_original, device)
        _embeds_soft = get_prompt_embeds(pipe, prompts_soft, device)

        embeds_original.append(_embeds_original)
        embeds_soft.append(_embeds_soft)

    embeds_original = torch.stack(
        embeds_original
    )  # size = [n_temp, n_cls, n_seq, f_dim]
    embeds_soft = torch.stack(embeds_soft)

    return embeds_original, embeds_soft


class GenerateImage:
    def __init__(
        self,
        pipe,
        device,
        mode,
        guidance_scale,
        num_inference_steps,
        n_img_per_class,
        save_dir,
        count_start,
        bs,
        n_shot,
        n_template,
        dataset,
    ):
        self.pipe = pipe
        self.device = device
        self.mode = mode
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.n_img_per_class = n_img_per_class
        self.save_dir = save_dir
        self.count_start = count_start
        self.bs = bs
        self.n_shot = n_shot
        self.n_template = n_template
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)

        self.resize_fn = tv.transforms.Resize(
            224, interpolation=tv.transforms.InterpolationMode.BICUBIC
        )

        self.run = self.name_template_method

    def update_pipe(self, pipe):
        # for datadream
        self.pipe = pipe

    def save_data(self, outputs, save_dir, count):
        images = outputs.images
        for image in images:
            fpath = ospj(save_dir, f"{count}.png")
            image = image.resize((512, 512))
            image.save(fpath)
            count += 1
        return count

    def run_pipe(self, prompts):
        if isinstance(prompts, list):
            prompt_embeds = None
        elif isinstance(prompts, torch.Tensor):
            prompt_embeds = prompts
            prompts = None

        lora_scale = 1 if self.mode == "datadream" else 0

        outputs = self.pipe(
            prompt=prompts,
            prompt_embeds=prompt_embeds,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
        )
        return outputs

    def set_save_dir(self, classname, prompts):
        save_dir = ospj(self.save_dir, "train", classname)

        make_dirs(save_dir)
        if isinstance(prompts, list):
            with open(ospj(save_dir, "prompts.json"), "w") as f:
                json.dump(prompts, f, indent=4)

        return save_dir

    def decorator_batch_prompts(prompt_fn):
        def wrapper(self, classname):
            # prompts for input to SD
            prompts = prompt_fn(self, classname)

            # make directory
            save_dir = self.set_save_dir(classname, prompts)

            count = self.count_start
            prompts = prompts[self.count_start :]

            for prompts_batch in batch_iteration(prompts, self.bs):
                # generate images
                outputs = self.run_pipe(prompts_batch)

                # save
                count = self.save_data(outputs, save_dir, count)

        return wrapper

    @decorator_batch_prompts
    def name_template_method(self, classname):
        templates = TEMPLATES_SMALL[: self.n_template]
        n_repeat = self.n_img_per_class // len(templates) + 1
        prompts = [
            template.format(self.dataset_name, classname)
            for _ in range(n_repeat)
            for template in templates
        ]
        prompts = prompts[: self.n_img_per_class]
        return prompts



def set_local(dataset):
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local


def main(
    seed=0,
    sd_version="sd2.1",
    mode="datadream",  # zeroshot, datadream
    guidance_scale=2.0,
    num_inference_steps=50,
    n_img_per_class=100,
    count_start=0,
    n_set_split=5,
    split_idx=0,
    bs=10,
    # few-shot
    n_shot=0,
    n_template=0,
    dataset="imagenet",
    fewshot_seed="seed0",  # best or seed{number}.
    datadream_lr: float = 1e-4,
    datadream_epoch: int = 200,
    datadream_train_text_encoder: bool = True,
    is_tqdm: bool = True,
    is_dataset_wise_model: bool = False,
):
    if isinstance(n_set_split, str):
        n_set_split = int(n_set_split)
    if isinstance(split_idx, str):
        split_idx = int(split_idx)
    if isinstance(n_shot, str):
        n_shot = int(n_shot)

    assert mode in ("zeroshot", "datadream"), "Wrong `mode` argument."
    assert mode == "datadream" and n_shot >= 1, \
           "`n_shot` should be integer when `mode` is datadream."
    if mode == "zeroshot":
        n_shot = 0

    # set local arguments
    args_local = set_local(dataset)
    model_dir = args_local["model_dir"][sd_version]
    datadream_dir = args_local["datadream_dir"] if mode == "datadream" else None

    # save directory
    mid_dir = f"gs{guidance_scale}_nis{num_inference_steps}"
    if mode == "zeroshot":
        mid2_dir = f"shot{n_shot}_template{n_template}"
    elif mode == "datadream":
        mid2_dir = f"shot{n_shot}_{fewshot_seed}_template{n_template}"
        mid2_dir += f"_lr{datadream_lr}_ep{datadream_epoch}"
        if not datadream_train_text_encoder:
            mid2_dir += "_notextlora"
        if is_dataset_wise_model:
            mid2_dir += "_dswise"
    mid_dir = ospj(mid_dir, mid2_dir)
    save_dir = ospj(
        args_local["save_dir"],
        dataset,
        sd_version,
        mid_dir,
    )
    print(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load SD pipeline
    pipe = get_pipe(sd_version, model_dir, device, is_tqdm)
    if mode == "datadream":
        if is_dataset_wise_model:
            classname = "dataset-wise"
            pipe = update_pipe(
                pipe,
                n_shot,
                n_template,
                dataset,
                datadream_dir,
                datadream_lr,
                datadream_epoch,
                datadream_train_text_encoder,
                fewshot_seed,
                classname,
            )
        else:
            # update pipe in every class
            pass

    # load instance
    generate_image = GenerateImage(
        pipe=pipe,
        device=device,
        mode=mode,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        n_img_per_class=n_img_per_class,
        save_dir=save_dir,
        count_start=count_start,
        bs=bs,
        n_shot=n_shot,
        n_template=n_template,
        dataset=dataset,
    )

    iters = SUBSET_NAMES[dataset]

    # parallel computing
    step = len(iters) // n_set_split
    start_idx = split_idx * step
    end_idx = (split_idx + 1) * step if (split_idx + 1) != n_set_split else len(iters)
    print(
        f"SPLIT!! Out of {len(SUBSET_NAMES[dataset])} pairs, we generate from idx {start_idx} to {end_idx}."
    )
    iters_partial = iters[start_idx:end_idx]

    # generate & save synthetic images
    for classname in tqdm(iters_partial, total=len(iters_partial)):
        if mode == "datadream":
            if is_dataset_wise_model:
                # update pipe just in the beginning
                pass
            else:
                # update pipe
                pipe = get_pipe(sd_version, model_dir, device, is_tqdm)
                pipe = update_pipe(
                    pipe,
                    n_shot,
                    n_template,
                    dataset,
                    datadream_dir,
                    datadream_lr,
                    datadream_epoch,
                    datadream_train_text_encoder,
                    fewshot_seed,
                    classname,
                )
                generate_image.update_pipe(pipe)

        # run
        set_seed(seed)
        generate_image.run(classname)


if __name__ == "__main__":
    fire.Fire(main)
