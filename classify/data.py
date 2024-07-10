import os
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from collections import defaultdict

from utils import make_dirs
from util_data import (
    SUBSET_NAMES,
    configure_metadata, get_image_ids, get_class_labels,
    GaussianBlur, Solarization,
)

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(model_type):

    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    elif model_type == "resnet50":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

    aux_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomApply(
            [
                tv.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        tv.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])
    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.RandAugment(),
        tv.transforms.RandomResizedCrop(
            224, 
            scale=(0.25, 1.0), 
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
            antialias=None,
        ),
        aux_transform,
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.Resize(
            224, 
            interpolation=tv.transforms.functional.InterpolationMode.BICUBIC
        ),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    return train_transform, test_transform


class ImageNetDatasetFromMetadata(Dataset):
    def __init__(
        self, 
        data_root, 
        metadata_root, 
        transform, 
        proxy, 
        target_label=None, 
        n_img_per_cls=None,
        dataset="imagenet",
        n_shot=0,
        real_train_fewshot_data_dir='',
        is_pooled_fewshot=False,
    ):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.is_pooled_fewshot = is_pooled_fewshot
        
        if not is_pooled_fewshot:
            """ full data """
            if n_img_per_cls is not None:
                value_counts = defaultdict(int)

                tmp = {}
                for k, v in self.image_labels.items():
                    if value_counts[v] < n_img_per_cls:
                        tmp[k] = v
                        value_counts[v] += 1
                self.image_labels = tmp

            if target_label is not None:
                self.image_labels = {k: v for k, v in self.image_labels.items() 
                                     if v == target_label}

            self.image_ids = list(self.image_labels.keys())

        else:
            """ only fewshot data """
            self.image_paths = []
            self.image_labels = []
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_img_paths = os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name))
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir, 
                        class_name, 
                        real_img_paths[i]
                    ) for i in range(n_shot)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)


    def get_data(self, fpath):
        x = Image.open(fpath)
        x = x.convert('RGB')
        return x
            
    def __getitem__(self, idx):
        if not self.is_pooled_fewshot: # full data
            image_id = self.image_ids[idx]
            image = self.get_data(ospj(self.data_root, image_id))
            image_label = self.image_labels[image_id]
        else: # few-shot
            image_id = self.image_paths[idx]
            image = self.get_data(self.image_paths[idx])
            image_label = self.image_labels[idx]
        image = self.transform(image)
        return image, image_label

    def __len__(self):
        if not self.is_pooled_fewshot:
            return len(self.image_ids)
        else:
            return len(self.image_paths)


class DatasetSynthImage(Dataset):
    def __init__(
        self, 
        synth_train_data_dir, 
        transform, 
        target_label=None, 
        n_img_per_cls=None,
        dataset='imagenet', 
        n_shot=0,
        real_train_fewshot_data_dir='', 
        is_pooled_fewshot=False, 
        **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        self.transform = transform
        self.is_pooled_fewshot = is_pooled_fewshot
        
        self.image_paths = []
        self.image_labels = []

        value_counts = defaultdict(int)
        for label, class_name in enumerate(SUBSET_NAMES[dataset]):
            if target_label is not None and label != target_label:
                continue
            for fname in os.listdir(ospj(synth_train_data_dir, class_name)):
                if fname.endswith(".txt"):
                    continue
                if fname.endswith(".json"):
                    continue
                if n_img_per_cls is not None:
                    if value_counts[label] < n_img_per_cls:
                        value_counts[label] += 1
                    else:
                        continue
                self.image_paths.append(
                    ospj(synth_train_data_dir, class_name, fname))
                self.image_labels.append(label)

        if is_pooled_fewshot:
            if n_shot == 0:
                n_shot = 16
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_img_paths = os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name))
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir, 
                        class_name, 
                        real_img_paths[i]
                    ) for i in range(n_shot)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)
                
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        is_real = "real_train" in image_path

        elif self.is_pooled_fewshot:
            return image, image_label, is_real
        else:
            return image, image_label

    def __len__(self):
        return len(self.image_paths)


def filter_dset(dataset, n_img_per_cls, dataset_name):
    import random
    print(n_img_per_cls)
    if dataset_name == 'pets':
        _images = dataset._images
        _labels = dataset._labels
    elif dataset_name == 'stl10':
        _images = dataset.data
        _labels = dataset.labels
    elif dataset_name == 'food101' or dataset_name == 'fgvc_aircraft' or dataset_name == 'dtd' \
            or dataset_name == 'flowers102' or dataset_name == 'sun397':
        _images = dataset._image_files
        _labels = dataset._labels
    elif dataset_name == 'eurosat':
        _images = [sample[0] for sample in dataset.samples]
        _labels = [sample[1] for sample in dataset.samples]
    elif dataset_name == 'cars':
        _images = [sample[0] for sample in dataset._samples]
        _labels = [sample[1] for sample in dataset._samples]
    elif dataset_name == 'caltech101':
        _images = dataset.index
        _labels = dataset.y
    else:
        raise ValueError("Please specify valid dataset.")
    new_images = []
    new_labels = []
    for i in set(_labels):
        candidates = [j for j in range(len(_labels)) if _labels[j] == i]
        img_per_cls = min(n_img_per_cls, len(candidates))  # allow for less if not enoug
        idx = random.sample(range(0, len(candidates)), img_per_cls)
        new_images.extend([_images[candidates[j]] for j in idx])
        new_labels.extend([_labels[candidates[j]] for j in idx])
    if dataset_name == 'pets':
        dataset._images = new_images
        dataset._labels = new_labels
    elif dataset_name == 'stl10':
        import numpy as np
        dataset.data = np.asarray(new_images)
        dataset.labels = np.asarray(new_labels)
    elif dataset_name == 'food101' or dataset_name == 'fgvc_aircraft' or dataset_name == 'dtd'\
            or dataset_name == 'flowers102' or dataset_name == 'sun397':
        dataset._image_files = new_images
        dataset._labels = new_labels
    elif dataset_name == 'eurosat':
        dataset.samples = [(im, lab) for im, lab in zip(new_images, new_labels)]
        dataset.targets = new_labels
    elif dataset_name == 'cars':
        dataset._samples = [(im, lab) for im, lab in zip(new_images, new_labels)]
    elif dataset_name == 'caltech101':
        dataset.index = new_images
        dataset.y = new_labels
    else:
        raise ValueError("Please specify valid dataset.")
    return dataset


def split_eurosat(file_path, split, dataset):
    split_file_path = os.path.join(file_path, 'split_zhou_EuroSAT.json')
    if not os.path.exists(split_file_path):
        # split taken from https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#eurosat
        raise ValueError("Please download or copy split_zhou_EuroSAT.json into the dataset directory. (This can "
                         "also be found at /shared-local/jbader40/data/eurosat/EuroSAT/train/split_zhou_EuroSAT.json).")
    f = open(split_file_path)
    split_files = json.load(f)
    data = [os.path.join(file_path, 'eurosat', '2750', path[0]) for path in split_files[split]]
    dataset.samples = [sample for sample in dataset.samples if sample[0] in data]
    dataset.labels = [s[1] for s in dataset.samples]
    return dataset


def split_sun(file_path, split, dataset):
    import csv
    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/sun397/split_coop.csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(file_path, 'SUN397') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._image_files = [l for i, l in enumerate(dataset._image_files) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def split_caltech(file_path, split, dataset):
    import csv
    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    ind_to_keep = [i for i in range(len(dataset.index)) if
                   os.path.join(dataset.categories[dataset.y[i]],
                                'image_' + '{:04d}'.format(dataset.index[i]) +
                                '.jpg') in split_files]
    dataset.index = [dataset.index[i] for i in ind_to_keep]
    dataset.y = [dataset.y[i] for i in ind_to_keep]
    # shift everything from 2 up down by one, because faces_easy at idx=1 not used
    dataset.y = [i if i < 1 else i - 1 for i in dataset.y]
    # remove Faces_easy
    dataset.categories.remove("Faces_easy")
    dataset.annotation_categories.remove("Faces_3")
    return dataset


def split_dtd(real_train_data_dir, train_transform, split):
    import csv
    dtd_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='train',
        transform=train_transform,
        download=True,
    )
    val_dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='val',
        transform=train_transform,
        download=True,
    )
    test_dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='test',
        transform=train_transform,
        download=True,
    )
    train_dataset._image_files = train_dataset._image_files + val_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + val_dataset._labels + test_dataset._labels

    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(dtd_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(dtd_path_train, 'dtd', 'dtd', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(train_dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [l for i, l in enumerate(train_dataset._image_files) if i in ind_to_keep]
    train_dataset._labels = [l for i, l in enumerate(train_dataset._labels) if i in ind_to_keep]
    return train_dataset


def split_flowers(real_train_data_dir, train_transform, split):
    import csv
    flowers_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='train',
        transform=train_transform,
        download=True,
    )
    val_dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='val',
        transform=train_transform,
        download=True,
    )
    test_dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='test',
        transform=train_transform,
        download=True,
    )
    train_dataset._image_files = train_dataset._image_files + val_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + val_dataset._labels + test_dataset._labels

    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(flowers_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(flowers_path_train, 'flowers-102', 'jpg') + '/'
    ind_to_keep = [i for i, file in enumerate(train_dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [l for i, l in enumerate(train_dataset._image_files) if i in ind_to_keep]
    train_dataset._labels = [l for i, l in enumerate(train_dataset._labels) if i in ind_to_keep]
    return train_dataset


def split_food(real_train_data_dir, train_transform, split):
    import csv
    food_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.Food101(
        root=food_path_train,
        split='train',
        transform=train_transform,
        download=True,
    )
    test_dataset = tv.datasets.Food101(
        root=food_path_train,
        split='test',
        transform=train_transform,
        download=True,
    )
    train_dataset._image_files = train_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + test_dataset._labels

    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(food_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(food_path_train, 'food-101', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(train_dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [l for i, l in enumerate(train_dataset._image_files) if i in ind_to_keep]
    train_dataset._labels = [l for i, l in enumerate(train_dataset._labels) if i in ind_to_keep]
    return train_dataset


def split_pets(real_train_data_dir, train_transform, split):
    import csv
    pets_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='trainval',
        target_types='category',
        download=True,
        transform=train_transform,
    )
    test_dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='test',
        target_types='category',
        download=True,
        transform=train_transform,
    )
    train_dataset._images = train_dataset._images + test_dataset._images
    train_dataset._labels = train_dataset._labels + test_dataset._labels

    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(pets_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'].split('/')[-1])
    file_path_full = os.path.join(pets_path_train, 'oxford-iiit-pet', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(train_dataset._images)
                   if str(file).replace(file_path_full, '') in split_files]
    train_dataset._images = [l for i, l in enumerate(train_dataset._images) if i in ind_to_keep]
    train_dataset._labels = [l for i, l in enumerate(train_dataset._labels) if i in ind_to_keep]
    return train_dataset


def get_data_loader(
    real_train_data_dir="",
    real_test_data_dir="",
    metadata_dir="metadata",
    dataset="imagenet", 
    bs=32, 
    eval_bs=32,
    is_rand_aug=True,
    target_label=None,
    n_img_per_cls=None,
    is_synth_train=False,
    n_shot=0,
    real_train_fewshot_data_dir='',
    is_pooled_fewshot=False,
    model_type=None,
):

    train_transform, test_transform = get_transforms(model_type)

    if is_synth_train:
        train_loader = None
    else:
        if dataset == 'imagenet':
            train_dataset = ImageNetDatasetFromMetadata(
                data_root=real_train_data_dir,
                metadata_root=ospj(metadata_dir, 'train'),
                transform=train_transform if is_rand_aug else test_transform,
                proxy=False,
                target_label=target_label,
                n_img_per_cls=n_img_per_cls,
                dataset=dataset,
                n_shot=n_shot,
                real_train_fewshot_data_dir=real_train_fewshot_data_dir,
                is_pooled_fewshot=is_pooled_fewshot,
            )
        elif dataset == 'pets':
            train_dataset = split_pets(real_train_data_dir, train_transform, 'train')

            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'stl10':
            stl10_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.STL10(
                root=stl10_path_train,
                split='train',
                download=True,
                transform=train_transform,
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'food101':
            train_dataset = split_food(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'fgvc_aircraft':
            aircraft_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.FGVCAircraft(
                root=aircraft_path_train,
                split='trainval',
                annotation_level='variant',
                transform=train_transform,
                download=True,
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'eurosat':
            eurosat_path_train = os.path.join(real_train_data_dir, 'train')
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            train_dataset = tv.datasets.EuroSAT(
                root=eurosat_path_train,
                transform=train_transform,
                download=True,
            )
            train_dataset = split_eurosat(eurosat_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'cars':
            cars_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.StanfordCars(
                root=cars_path_train,
                split='train',
                transform=train_transform,
                download=False,
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'dtd':
            train_dataset = split_dtd(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'flowers102':
            train_dataset = split_flowers(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'sun397':
            sun_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.SUN397(
                root=sun_path_train,
                transform=train_transform,
                download=True,
            )
            train_dataset = split_sun(sun_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'caltech101':
            caltech_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.Caltech101(
                root=caltech_path_train,
                transform=train_transform,
                download=True,
            )
            train_dataset = split_caltech(caltech_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls,
                                        dataset_name=dataset)
        else:
            raise ValueError("Please specify a valid dataset.")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, 
            sampler=None,
            shuffle=is_rand_aug,
            prefetch_factor=4, pin_memory=True,
            num_workers=16)

#####################################################
#####################################################
#####################################################

    if dataset == 'imagenet':
        test_dataset = DatasetFromMetadata(
            data_root=real_test_data_dir,
            metadata_root=ospj(metadata_dir, 'test'),
            transform=test_transform,
            proxy=False,
            target_label=target_label,
            dataset=dataset,
        )
    # TODO: update arguments
    elif dataset == 'pets':
        test_dataset = split_pets(real_train_data_dir, test_transform, 'test')
    elif dataset == 'stl10':
        stl10_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.STL10(
            root=stl10_path_test,
            split='test',
            download=True,
            transform=test_transform,
        )
    elif dataset == 'food101':
        test_dataset = split_food(real_train_data_dir, test_transform, 'test')
    elif dataset == 'fgvc_aircraft':
        aircraft_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.FGVCAircraft(
            root=aircraft_path_test,
            split='test',
            annotation_level='variant',
            transform=test_transform,
            download=True,
        )
    elif dataset == 'eurosat':
        eurosat_path_test = os.path.join(real_train_data_dir, 'train')
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        test_dataset = tv.datasets.EuroSAT(
            root=eurosat_path_test,
            transform=test_transform,
            download=True,
        )
        test_dataset = split_eurosat(eurosat_path_test, 'train', test_dataset)
    elif dataset == 'cars':
        # note: this must be train, because the Cars dataset had to be downloaded by hand and is kept in the same
        # directory as the training data
        cars_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.StanfordCars(
            root=cars_path_test,
            split='test',
            transform=test_transform,
            download=False,
        )
    elif dataset == 'dtd':
        test_dataset = split_dtd(real_train_data_dir, test_transform, 'train')
    elif dataset == 'flowers102':
        test_dataset = split_flowers(real_train_data_dir, test_transform, 'train')
    elif dataset == 'sun397':
        sun_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.SUN397(
            root=sun_path_test,
            transform=test_transform,
            download=True,
        )
        test_dataset = split_sun(sun_path_test, 'test', test_dataset)
    elif dataset == 'caltech101':
        caltech_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.Caltech101(
            root=caltech_path_test,
            transform=test_transform,
            download=True,
        )
        test_dataset = split_caltech(caltech_path_test, 'test', test_dataset)
    else:
        raise ValueError("Please specify a valid dataset.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_bs, shuffle=False, 
        num_workers=16, pin_memory=True)

    return train_loader, test_loader


def get_synth_train_data_loader(
    synth_train_data_dir="data_synth",
    bs=32, 
    is_rand_aug=True,
    target_label=None,
    n_img_per_cls=None,
    dataset='imagenet',
    n_shot=0,
    real_train_fewshot_data_dir='',
    is_pooled_fewshot=False,
    model_type=None,
):

    train_transform, test_transform = get_transforms(model_type)

    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir, 
        transform=train_transform if is_rand_aug else test_transform,
        target_label=target_label,
        n_img_per_cls=n_img_per_cls,
        dataset=dataset,
        n_shot=n_shot,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_pooled_fewshot=is_pooled_fewshot,
    ) 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, 
        sampler=None,
        shuffle=is_rand_aug,
        num_workers=16, pin_memory=True,
    )
    return train_loader



