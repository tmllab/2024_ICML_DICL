from torch.utils.data import DataLoader
from mydatasets.image_text_cifar import image_text_MNIST
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def get_mnist(args, processor, return_classnames=False):
    image_text_dataset = image_text_MNIST(args.data_location, train=True, transform=processor)
    sampler = RandomSampler(image_text_dataset, replacement=True, num_samples=image_text_dataset.__len__())
    image_text_dataloader = DataLoader(image_text_dataset,
                                    shuffle=False,
                                    sampler=sampler,
                                    batch_size=args.batch_size,
                                )
    
    return_list = []
    if args.exemplar:
        exemplar_dataset = image_text_MNIST(args.data_location, train=True, transform=processor, exemplar=True, num_exemplar=args.num_exemplar)
        exemplar_sampler = SequentialSampler(exemplar_dataset)
        exemplar_dataloader = DataLoader(exemplar_dataset,
                                        shuffle=False,
                                        sampler=exemplar_sampler,
                                        batch_size=args.batch_size,
                                    )
        return_list = [image_text_dataloader, exemplar_dataloader, image_text_dataset.num_classes]
    else:
        return_list = [image_text_dataloader, image_text_dataset.num_classes]

    if return_classnames:
        return_list.append(list(image_text_dataset.label_to_class_mapping.values()))

    return tuple(return_list)

