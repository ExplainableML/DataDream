#### In this step, we finetune a Stable Diffusion model with few-shots.

## Setup

Fill in the paths in `local.yaml` file. Users can also add datasets in the key `fewshot_data_dir`. 

## Running

```python
CUDA_VISIBLE_DEVICES=$GPU, accelerate launch datadream.py \
--dataset=$DATASET \
--target_class_idx=$CLASS_IDX \
--fewshot_seed=seed0 \
--n_shot=16 \
--n_template=1 \
--train_text_encoder=True \
--resume_from_checkpoint=None \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="cosine" \
--lr_warmup_steps=100 \
--num_train_epochs=200 \
--report_to="tensorboard" \
--is_tqdm=True \
--output_dir=outputs
```

where `DATASET` is the dataset name and `CLASS_IDX` is an integer number. You could alternatively run

```python
bash bash_run.sh $GPU $SPLIT_IDX
```
where `SPLIT_IDX` is an integer number that specifies the list of integer numbers for `CLASS_IDX`.
