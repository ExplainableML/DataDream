
import sys
import logging
import random
import atexit
import getpass
import shutil
import time
import os
import yaml
import json
import argparse
from os.path import join as ospj

from util_data import SUBSET_NAMES

_MODEL_TYPE = ("resnet50", "clip")


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v

def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)

def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)

def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs


def set_local(args):
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)

    args.real_train_data_dir = args_local["real_train_data_dir"][args.dataset]
    args.real_train_fewshot_data_dir = ospj(
        args_local["real_train_fewshot_data_dir"][args.dataset], 
        args.fewshot_seed
    )
    args.real_test_data_dir = args_local["real_test_data_dir"][args.dataset]
    args.synth_train_data_dir = args_local["synth_train_data_dir"]
    args.metadata_dir = args_local["metadata_dir"]
    args.clip_download_dir = args_local["clip_download_dir"]
    args.wandb_key = args_local["wandb_key"]


def set_output_dir(args):
    n_img_per_cls = "full" if args.n_img_per_cls is None else args.n_img_per_cls
    mid2 = f"n_img_per_cls_{n_img_per_cls}"
    if args.is_synth_train:
        mid3 = args.sd_version
    else:
        mid3 = "baseline"
        if args.is_pooled_fewshot:
            mid3 += f"_shot{args.n_shot}_{args.fewshot_seed}"
    if args.is_synth_train:
        if args.n_shot == 0: # zeroshot
            mid3 = ospj(mid3, f"shot{args.n_shot}_template{args.n_template}")
        else: # datadream
            mid3 = ospj(mid3, f"shot{args.n_shot}_{args.fewshot_seed}_template{args.n_template}")
            mid3 += f"_ddlr{args.datadream_lr}"
            mid3 += f"_ddep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                mid3 += "_notextlora"
            if args.is_dataset_wise:
                mid3 += "_dswise"
        if args.is_pooled_fewshot:
            mid3 += f"_lbd{args.lambda_1}"
    mixaug = "_mixuag" if args.is_mix_aug else ""
    mid4 = f"lr{args.lr}_wd{args.wd}{mixaug}"

    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version

    args.output_dir = ospj(args.output_dir, args.dataset, model_type, mid2, mid3, mid4)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


def set_synth_train_data_dir(args):
    if args.is_synth_train: 
        if args.n_shot == 0:
            mid_dir = f"shot{args.n_shot}_template{args.n_template}"
        else:
            mid_dir = f"shot{args.n_shot}_{args.fewshot_seed}_template{args.n_template}"
            mid_dir += f"_lr{args.datadream_lr}"
            mid_dir += f"_ep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                mid_dir += "_notextlora"
            if args.is_dataset_wise:
                mid_dir += "_dswise"
        args.synth_train_data_dir = ospj(
            args.synth_train_data_dir,
            args.dataset,
            args.sd_version, 
            f"gs{args.guidance_scale}_nis{args.num_inference_steps}",
            mid_dir, 
            "train",
        ) 


def set_log(output_dir):
    log_file_name = ospj(output_dir, 'log.log')
    Logger(log_file_name)


def set_wandb_group(args):
    pooled = f"pool_lbd{args.lambda_1}" if args.is_pooled_fewshot else ""
    mixaug = "_mixaug" if args.is_mix_aug else ""
    synth_setting = ""
    if args.is_synth_train: 
        if args.n_shot == 0:
            synth_setting = "zeroshot"
        else:
            synth_setting = "datadream"
            synth_setting += f"_ddlr{args.datadream_lr}"
            synth_setting += f"_ddep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                synth_setting += "_notl"
            if args.is_dataset_wise:
                synth_setting += "_dswise"
    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version
    args.wandb_group = f"{args.dataset[:4]}_{model_type}_{pooled}_shot{args.n_shot}_{args.fewshot_seed}_{synth_setting}_gs{args.guidance_scale}_nipc{args.n_img_per_cls}_lr{args.lr}_wd{args.wd}{mixaug}"


def set_follow_up_configs(args):
    set_output_dir(args)
    set_synth_train_data_dir(args)
    set_log(args.output_dir)
    if args.wandb_group is None:
        set_wandb_group(args)

def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')

    # CLIP setting
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)


    # Data
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument("--n_img_per_cls", type=int2none, default=100)
    parser.add_argument("--is_mix_aug", type=str2bool, default=False,
                        help="use mixup and cutmix")
    parser.add_argument("--is_pooled_fewshot", type=str2bool, default=False)
    parser.add_argument("--lambda_1", type=float2none, default=0,
                        help="weight for loss from real/synth data")
    parser.add_argument("--fewshot_seed", type=str2none, default="seed0",
                        help="best or seed{number}.")
    parser.add_argument("--is_dataset_wise", type=str2bool, default=False)
    parser.add_argument("--datadream_lr", type=float2none, default=1e-4)
    parser.add_argument("--datadream_epoch", type=int2none, default=200)
    parser.add_argument("--datadream_train_text_encoder", type=str2bool, default=True)

    # stable diffusion
    parser.add_argument("--is_synth_train", type=str2bool, default=False)
    parser.add_argument("--sd_version", type=str2none, default=None)
    parser.add_argument("--guidance_scale", type=float2none, default=2.0)
    parser.add_argument("--num_inference_steps", type=int2none, default=50)
    # for few-shot
    parser.add_argument("--n_shot", type=int2none, default=16)
    parser.add_argument("--n_template", type=int2none, default=1)


    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size_eval",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float2none,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float2none,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=25,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=100,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )
    # wandb args
    parser.add_argument('--log', type=str, default='tensorboard', help='How to log')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='datadream', help='Wandb project name')
    parser.add_argument('--wandb_group', type=str2none, default=None, help='Name of the group for wandb runs')
    parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')

    args = parser.parse_args()

    set_local(args)
    set_follow_up_configs(args)

    return args


