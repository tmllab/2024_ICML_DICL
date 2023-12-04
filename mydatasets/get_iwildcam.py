import ImageNet_datasets

split_name = ['IWildCamNonEmpty', 'IWildCamIDNonEmpty', 'IWildCamOODNonEmpty', 'IWildCam']
chosen_name = split_name[1]
def get_iwildcam(args, processor, return_classnames=False):
    dataset_class = getattr(ImageNet_datasets, chosen_name)
    image_text_dataset = dataset_class(processor,
                            location=args.data_location,
                            remove_non_empty=True,
                            batch_size=args.batch_size)
    print(chosen_name, ' number of examples: ', len(image_text_dataset.train_dataset.indices))
    image_text_dataloader = image_text_dataset.train_loader
    n_classes = image_text_dataset.train_dataset._n_classes
    
    return_list = []
    if args.exemplar:
        exemplar_dataset = dataset_class(processor,
                                location=args.data_location,
                                remove_non_empty=True,
                                batch_size=args.batch_size, 
                                exemplar=args.exemplar, 
                                num_exemplar=args.num_exemplar)
        print(chosen_name, ' number of exemplars: ', len(exemplar_dataset.train_dataset.indices))
        exemplar_dataloader = exemplar_dataset.train_loader
        return_list = [image_text_dataloader, exemplar_dataloader, n_classes]
    else:
        return_list = [image_text_dataloader, n_classes]

    if return_classnames:
        return_list.append(image_text_dataset.classnames)

    return tuple(return_list)


