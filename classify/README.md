#### We assume a synthetic dataset is generated. In this step, we train the classifier, either using solely the synthetic data or using both synthetic and few-shot real data.

## Setup

Fill in the paths in `local.yaml` file. User can also add dataset names and their paths in keys `real_train_data_dir`, `real_test_data_dir`, and `real_train_fewshot_data_dir`.

## Running

```python
CUDA_VISIBLE_DEVICES=$GPU python main.py \
--model_type=clip \
--dataset=$DATASET \
--is_synth_train=True \
--is_pooled_fewshot=$IS_POOLED \
--is_dataset_wise=$IS_DATASET_WISE \
--datadream_lr=1e-4 \
--datadream_epoch=200 \
--n_shot=16 \
--fewshot_seed=seed0 \
--n_img_per_cls=500 \
--sd_version="sd2.1" \
--n_template=1 \
--guidance_scale=2.0 \
--lambda_1=0.8 \
--epochs=$EPOCH \
--warmup_epochs=$WARMUP_EPOCH \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--is_mix_aug=True \
--log=wandb
```

where `DATASET` is the dataset name. We run the DataDream_cls when `IS_DATASETWISE=False` and DataDream_dset when `IS_DATASETWISE=True`. The boolean variable `IS_POOLED` indicates whether to contain real few-shot data in the training set. You could alternatively run

```python
bash bash_run.sh
```
