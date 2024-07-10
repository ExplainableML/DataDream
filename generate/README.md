#### We assume to have DataDream checkpoints. In this step, we generate synthetic images with DataDream model.

## Setup

Fill in the paths in `local.yaml` file.  

## Running

```python
CCUDA_VISIBLE_DEVICES=$GPU python generate.py \
--sd_version=sd2.1 \
--mode=datadream \
--is_dataset_wise_model=$IS_DATASETWISE \
--dataset=$DATASET \
--n_shot=16 \
--fewshot_seed=seed0 \
--bs=10 \
--n_img_per_class=500 \
--guidance_scale=2.0 \
--n_template=1 \
--datadream_lr=1e-4 \
--datadream_epoch=200 \
--n_set_split=$N_SET_SPLIT \
--split_idx=$SPLIT_IDX
```

where `DATASET` is the dataset name. We run the DataDream_cls when `IS_DATASETWISE=False` and DataDream_dset when `IS_DATASETWISE=True`. Two variables `N_SET_SPLIT` and `SPLIT_IDX` are integer numbers that specify the list of classes of which the synthetic images will be generated. You could alternatively run

```python
bash bash_run.sh $GPU $SPLIT_IDX
```
