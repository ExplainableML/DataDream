

import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj
import wandb

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.transforms import v2

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    MetricLogger,
)

from config import get_args
from data import get_data_loader, get_synth_train_data_loader
from models.clip import CLIP
from models.resnet50 import ResNet50
from util_data import SUBSET_NAMES



def load_data_loader(args):
    train_loader, test_loader = get_data_loader(
        real_train_data_dir=args.real_train_data_dir,
        real_test_data_dir=args.real_test_data_dir,
        metadata_dir=args.metadata_dir,
        dataset=args.dataset, 
        bs=args.batch_size,
        eval_bs=args.batch_size_eval,
        n_img_per_cls=args.n_img_per_cls,
        is_synth_train=args.is_synth_train,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
    )
    return train_loader, test_loader



def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        bs=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        dataset=args.dataset,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
    )
    return synth_train_loader


def main(args):
    args.n_classes = len(SUBSET_NAMES[args.dataset])

    os.makedirs(args.output_dir, exist_ok=True)

    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data loader
    # ==================================================
    train_loader, val_loader = load_data_loader(args)
    if args.is_synth_train:
        train_loader = load_synth_train_data_loader(args)

        
    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":
        # TODO
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50": 
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    scheduler = None
    optimizer = torch.optim.AdamW(
        params_groups, lr=args.lr, weight_decay=args.wd,
    )
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            project=args.wandb_project, 
            group=args.wandb_group, 
            name=args.wandb_group,
            settings=wandb.Settings(start_method='fork'),
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()
    elif args.log == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb-{}".format(args.local_rank))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir, flush_secs=30)

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_top1 = 0.

    for epoch in range(0, args.epochs):
        train_stats, best_stats, best_top1 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
            val_loader, best_stats, best_top1, 
        )

#         if args.dataset in ("imagenet", "sun397"):
#             # evaluate ten times in each epoch
#             # here we only save train stats
#             if args.log == 'wandb':
#                 train_stats.update({"epoch": epoch})
#                 wandb.log(train_stats)
#         else:
        # ============ evaluate model ... ============
        test_stats = eval(
            model, criterion, val_loader, epoch, fp16_scaler, args)

        # ============ saving logs and model checkpoint ... ============
        if test_stats["test/top1"] > best_top1:
            best_top1 = test_stats["test/top1"]
            best_stats = test_stats
            save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")

        if epoch + 1 == args.epochs:
            test_stats['test/best_top1'] = best_stats["test/top1"]
            test_stats['test/best_loss'] = best_stats["test/loss"]

        if args.log == 'wandb':
            train_stats.update({"epoch": epoch})
            wandb.log(train_stats)
            wandb.log(test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        if args.is_synth_train and args.is_pooled_fewshot:
            image, label, is_real = batch
        else:
            image, label = batch

        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)

        # apply CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p >= 0.2:
                pass
            else:
                if args.is_synth_train and args.is_pooled_fewshot:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    image_real, label_real = image[is_real==1], label[is_real==1]
                    image_synth, label_synth = image[is_real==0], label[is_real==0]

                    image_real, label_real = cutmix_or_mixup(image_real, label_real)
                    image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)

                    new_image[is_real==1] = image_real
                    new_image[is_real==0] = image_synth
                    new_label[is_real==1] = label_real
                    new_label[is_real==0] = label_synth

                    image = new_image
                    label = new_label

                else:
                    image, label = cutmix_or_mixup(image, label)
            

        it = len(data_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(image)

            if args.is_synth_train and args.is_pooled_fewshot:
                loss_real = criterion(logit[is_real == 1], label[is_real == 1])
                loss_synth = criterion(logit[is_real == 0], label[is_real == 0])
                loss = args.lambda_1 * loss_real + (1 - args.lambda_1) * loss_synth
            else:
                loss = criterion(logit, label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

#         # eval in the middle if there's too much iterations
#         if args.dataset in ("imagenet", "sun397") and it % (len(data_loader) // 10) == 0:
#             test_stats = eval(
#                 model, criterion, val_loader, epoch, fp16_scaler, args)
#             if test_stats["test/top1"] > best_top1:
#                 best_top1 = test_stats["test/top1"]
#                 best_stats = test_stats
#                 save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")
#             if epoch + 1 == args.epochs:
#                 test_stats['test/best_top1'] = best_stats["test/top1"]
#                 test_stats['test/best_loss'] = best_stats["test/loss"]
#             if args.log == 'wandb':
#                 wandb.log(test_stats)
#             model.train()

#         if it % len(data_loader) == 5:
#             break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)

    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


@torch.no_grad()
def eval(model, criterion, data_loader, epoch, fp16_scaler, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    is_last = epoch + 1 == args.epochs
    if is_last:
        targets = []
        outputs = []

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):

        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image, phase="eval")
            loss = criterion(output, label)

        acc1, acc5 = get_accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

        if is_last:
            targets.append(label)
            outputs.append(output)

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)

    stat_dict = {"test/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}

    if is_last:
        targets = torch.cat(targets)
        outputs = torch.cat(outputs)

        # calculate per class accuracy
        acc_per_class = [
            get_accuracy(outputs[targets == cls_idx], targets[targets == cls_idx], topk=(1,))[0].item() 
            for cls_idx in range(args.n_classes)
        ]
        for cls_idx, acc in enumerate(acc_per_class):
            print("{} [{}]: {}".format(SUBSET_NAMES[args.dataset][cls_idx], cls_idx, str(acc)))
            stat_dict[SUBSET_NAMES[args.dataset][cls_idx] + '_cls-acc'] = acc

    return stat_dict


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]



def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(traceback.format_exc())
