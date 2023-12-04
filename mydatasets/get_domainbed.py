from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import DomainBed_datasets.domainbed.datasets as dbdatasets

import DomainBed_datasets.domainbed.lib.misc as misc
from DomainBed_datasets.domainbed.lib.fast_data_loader import InfiniteDataLoader
import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from mydatasets.utils import accuracy, AverageMeter, ProgressMeter, ForeverDataIterator


def get_domainbed(args, processor, return_classnames=True, task="domain_shift"):
    if task == "domain_shift":
        exemplar_datasets, val_datasets, test_datasets, class_names = \
            get_domainbed_datasets(dataset_name=args.chosen_name, root=args.data_location, processor=processor, targets=args.targets, holdout=0.2, num_exampar=args.num_exemplar)
        train_class_names = class_names
        exemplar_loader = DataLoader(ConcatDataset(exemplar_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print('exemplar number', len(exemplar_loader) * args.batch_size)
        print('val number', len(val_loader) * args.batch_size)
        template = "a photo of a {}."

    elif task == "in_the_wild":
        exemplar_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            get_domainbed_datasets(dataset_name=args.chosen_name, root=args.data_location, processor=processor, targets=args.targets, holdout=0.2, seed=args.seed, open_ratio=0.5)
        train_class_names = base_class_names
        exemplar_loader = DataLoader(ConcatDataset(exemplar_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
    
    return val_loader, exemplar_loader, len(train_class_names), train_class_names


def get_subdatasets(dataset, class_keys):
    base_keys, open_keys = [], []
    for i, (_, label) in enumerate(dataset.samples):
        if label in class_keys:
            base_keys.append(i)
        else:
            open_keys.append(i)
    base_dataset = misc._SplitDataset(dataset, base_keys)
    open_dataset = misc._SplitDataset(dataset, open_keys)
    return base_dataset, open_dataset


def get_exemplar_subdatasets(dataset, num_exemplar):
    original_dataset = dataset.underlying_dataset
    original_target_array = np.array(original_dataset.targets)
    target_array = original_target_array[dataset.keys]
    keys_array = np.array(dataset.keys)
    idxs = np.arange(len(target_array), dtype=int)
    exemplar_idxs = []
    for i in range(len(original_dataset.classes)):
        cls_idx = target_array == i
        try:
            select_fnames = np.random.choice(keys_array[cls_idx], size=num_exemplar, replace=False)
        except:
            print('Num exemplar more than Existing examples')
            print(i, keys_array, cls_idx)
            select_fnames = np.random.choice(keys_array[cls_idx], size=num_exemplar, replace=True)
        exemplar_idxs.extend(list(select_fnames))
    print('Number of exemplars: ', len(exemplar_idxs))
    dataset.keys = exemplar_idxs
    return dataset


def get_domainbed_datasets(dataset_name, root, processor, targets, holdout=0.2, open_ratio=0, num_exampar=0):
    assert dataset_name in vars(dbdatasets)
    hparams = {"data_augmentation": False}
    if "Spawrious" in dataset_name:
        datasets = vars(dbdatasets)[dataset_name](root, processor, targets, hparams, num_exemplar=num_exampar)
        train_datasets, test_datasets = datasets.train_datasets, datasets.test_datasets
        class_names = datasets.class_list
        return train_datasets, test_datasets, None, class_names
    else:
        datasets = vars(dbdatasets)[dataset_name](root, processor, targets, hparams)
        class_names = datasets[0].classes
        if open_ratio > 0:
            # Sample subclasses
            keys = list(range(len(class_names)))
            base_class_keys = keys[:int((1 - open_ratio) * len(keys))]
            base_class_names = [class_name for i, class_name in enumerate(class_names) if i in base_class_keys]
            open_class_names = [class_name for class_name in class_names if class_name not in base_class_names]
            in_bases, in_opens, out_bases, out_opens = [], [], [], []
            for env_i, env in enumerate(datasets):
                base_env, open_env = get_subdatasets(env, base_class_keys)
                out_base, in_base = misc.split_dataset(base_env, int(len(base_env) * holdout), misc.seed_hash(0, env_i, "base"))
                out_open, in_open = misc.split_dataset(open_env, int(len(open_env) * holdout), misc.seed_hash(0, env_i, "open"))
                in_bases.append(in_base)
                in_opens.append(in_open)
                out_bases.append(out_base)
                out_opens.append(out_open)
            train_datasets = [d for (i, d) in enumerate(in_bases) if i not in targets]
            val_datasets = [d for (i, d) in enumerate(out_bases) if i not in targets]
            test_datasets = [d for (i, d) in enumerate(in_bases) if i in targets] + [d for (i, d) in enumerate(out_bases) if i in targets]
            open_datasets = [d for (i, d) in enumerate(in_opens) if i in targets] + [d for (i, d) in enumerate(out_opens) if i in targets]
            return train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names
        else:
            in_splits, out_splits = [], []
            for env_i, env in enumerate(datasets):
                out, in_ = misc.split_dataset(env,
                    int(len(env) * holdout),
                    misc.seed_hash(0, env_i))
                in_splits.append(in_)
                out_splits.append(out)

            train_datasets = [d for (i, d) in enumerate(in_splits) if i not in targets]
            val_datasets = [d for (i, d) in enumerate(out_splits) if i not in targets]
            test_datasets = [d for (i, d) in enumerate(out_splits) if i in targets]

            num_env = len(train_datasets)
            exemplar_per_env = num_exampar // num_env
            exemplar_datasets = []
            for env_i, env in enumerate(train_datasets):
                exemplar_env = get_exemplar_subdatasets(env, exemplar_per_env)
                exemplar_datasets.append(exemplar_env)
            return exemplar_datasets, val_datasets, test_datasets, class_names

def get_forever_iter(datasets, batch_size, num_workers):
    iters = [InfiniteDataLoader(dataset, None, batch_size, num_workers) for dataset in datasets]
    return zip(*iters)




CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

