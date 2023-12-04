import wilds
from wilds.common.grouper import CombinatorialGrouper
from collections import defaultdict
from wilds.common.data_loaders import get_train_loader, get_eval_loader

def get_celebA(args, processor, return_classnames=False):
    # WILDS dataset
    full_dataset = wilds.get_dataset(
        dataset=args.dataset,
        root_dir=args.data_location,
        target_name=args.target_attribute)
    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=args.groupby_fields
    )
    split = args.split
    # Configure labeled torch datasets (WILDS dataset splits)
    wilds_datasets = defaultdict(dict)
    if split=='train':
        verbose = True
    elif split == 'val':
        verbose = True
    else:
        verbose = False
    # Get subset
    wilds_datasets[split]['dataset'] = full_dataset.get_subset(
        split,
        transform=processor)

    if split == 'train':
        wilds_datasets[split]['loader'] = get_train_loader(
            loader='group',
            dataset=wilds_datasets[split]['dataset'],
            batch_size=args.batch_size,
            uniform_over_groups=True,
            grouper=train_grouper,
            distinct_groups=False,
            n_groups_per_batch=args.batch_size,)
    else:
        wilds_datasets[split]['loader'] = get_eval_loader(
            loader='group',
            dataset=wilds_datasets[split]['dataset'],
            batch_size=args.batch_size,
            uniform_over_groups=True,
            grouper=train_grouper,
            distinct_groups=False,
            n_groups_per_batch=args.batch_size,)
    # Set fields
    wilds_datasets[split]['split'] = split
    wilds_datasets[split]['name'] = full_dataset.split_names[split]
    wilds_datasets[split]['verbose'] = verbose

    image_text_dataloader = wilds_datasets[split]['loader']

    if args.exemplar:
        wilds_datasets[split]['exemplar_dataset'] = full_dataset.get_exemplar_set(
            split,
            transform=processor,
            exemplar=args.exemplar,
            num_exemplar=args.num_exemplar)
        
        exemplar_dataloader = get_eval_loader(
            loader='standard',
            dataset=wilds_datasets[split]['exemplar_dataset'],
            grouper=train_grouper,
            batch_size=args.batch_size,)
        return image_text_dataloader, exemplar_dataloader, full_dataset._n_classes
    else:
        return image_text_dataloader, full_dataset._n_classes