import ImageNet_datasets

def get_imagenet(args, processor, return_classnames=False):
    dataset_class = getattr(ImageNet_datasets, args.chosen_name)
    image_text_dataset = dataset_class(processor,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            dataset_type='test-5k')
    print(args.chosen_name, ' number of examples: ', image_text_dataset.test_dataset.__len__())
    image_text_dataloader = image_text_dataset.test_loader
    n_classes = image_text_dataset.n_classes

    return_list = []
    if args.exemplar:
        exemplar_dataset = dataset_class(processor,
                                location=args.data_location,
                                # remove_non_empty=True,
                                batch_size=args.batch_size, 
                                exemplar=args.exemplar, 
                                num_exemplar=args.num_exemplar,
                                dataset_type='exemplar')
        print(args.chosen_name, ' number of exemplars: ', exemplar_dataset.test_dataset.__len__())
        exemplar_dataloader = exemplar_dataset.test_loader
        return_list = [image_text_dataloader, exemplar_dataloader, n_classes]
    else:
        return_list = [image_text_dataloader, n_classes]

    if return_classnames:
        return_list.append(image_text_dataset.classnames)

    return tuple(return_list)


