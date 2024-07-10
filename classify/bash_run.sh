
GPU="0"
OUTPUT_DIR="outputs"

DATASET="eurosat"

NIPC=20
LR=1e-4
MIN_LR=1e-8
WD=1e-4
EPOCH=10
WARMUP_EPOCH=3
IS_MIX_AUG=False

FEWSHOT_SEED="seed0"
N_SHOT=16
N_TEMPLATE=1

IS_SYNTH_TRAIN=True
IS_DATASET_WISE=False
DD_LR=1e-4
DD_EP=20
DD_TTE=True

IS_POOLED=False
LAMBDA_1=0.8


CUDA_VISIBLE_DEVICES=$GPU python main.py \
--model_type=clip \
--output_dir=$OUTPUT_DIR \
--n_img_per_cls=$NIPC \
--is_lora_image=True \
--is_lora_text=True \
--is_synth_train=$IS_SYNTH_TRAIN \
--sd_version="sd2.1" \
--n_template=$N_TEMPLATE \
--guidance_scale=2.0 \
--is_pooled_fewshot=$IS_POOLED \
--lambda_1=$LAMBDA_1 \
--epochs=$EPOCH \
--warmup_epochs=$WARMUP_EPOCH \
--log=wandb \
--wandb_project=datadream \
--dataset=$DATASET \
--n_shot=$N_SHOT \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--fewshot_seed=$FEWSHOT_SEED \
--is_mix_aug=$IS_MIX_AUG \
--is_dataset_wise=$IS_DATASET_WISE \
--datadream_lr=$DD_LR \
--datadream_epoch=$DD_EP \
--datadream_train_text_encoder=$DD_TTE \
$PARAM
