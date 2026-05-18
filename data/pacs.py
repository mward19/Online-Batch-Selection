# PACS dataset from https://arxiv.org/pdf/1710.03077
# https://huggingface.co/datasets/flwrlabs/pacs

from collections import defaultdict

import numpy as np
from torchvision import transforms
from datasets import load_dataset # huggingface
import torch
from torch.utils.data import Dataset, random_split, Subset, ConcatDataset



from .data_utils.generate_noise import apply_or_generate_label_noise

pacs_domains = ['sketch', 'cartoon', 'art_painting', 'photo']

pacs_classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
pacs_templates = [
    'a {} of a {}.',
    'a blurry {} of a {}.',
    'a black and white {} of a {}.',
    'a low contrast {} of a {}.',
    'a high contrast {} of a {}.',
    'a bad {} of a {}.',
    'a good {} of a {}.',
    'a {} of a small {}.',
    'a {} of a big {}.',
    'a {} of the {}.',
    'a blurry {} of the {}.',
    'a black and white {} of the {}.',
    'a low contrast {} of the {}.',
    'a high contrast {} of the {}.',
    'a bad {} of the {}.',
    'a good {} of the {}.',
    'a {} of the small {}.',
    'a {} of the big {}.',
]

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        # Some dataset wrappers (e.g., Subset) may not expose .targets
        self.targets = getattr(dataset, 'targets', None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        input = data['image']
        target = data['label']
        if self.transform is not None:
            input = self.transform(input)

        return {
            'input': input,
            'target': target,
            'index': index
        }

def _build_test_loader(config, dst_test, test_transform):
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    return torch.utils.data.DataLoader(
        wrapped_dataset(dst_test, test_transform), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )


def _build_dataset_info(
        config, 
        logger, 
        dataset_name, 
        dst_train, 
        dst_test, 
        auxiliary_test_dsts,
        num_classes, 
        classes, 
        templates, 
        include_noise=False, 
        transform=None, 
        test_transform=None
    ):
    auxiliary_test_loaders = [
        _build_test_loader(config, dst, test_transform) 
        for dst in auxiliary_test_dsts
    ]
    payload = {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train, transform),
        'test_loader': _build_test_loader(config, dst_test, test_transform),
        'auxiliary_test_loaders': auxiliary_test_loaders,
        'num_train_samples': len(dst_train),
        'classes': classes,
        'template': templates,
    }
    if include_noise:
        payload.update(
            apply_or_generate_label_noise(
                dataset=dst_train,
                num_classes=num_classes,
                dataset_config=config['dataset'],
                logger=logger,
                dataset_name=dataset_name,
                seed=config.get('seed'),
            )
        )
        payload['train_dset'] = wrapped_dataset(dst_train)
    return payload


def _train_test_split(config, dst_all, keep_classes):
    train_ratio = config['dataset']['train_ratio']

    # Separate data indices into domains
    by_domain = defaultdict(list)

    for i, data in enumerate(dst_all):
        if data['label'] not in keep_classes:
            continue
        by_domain[data["domain"]].append(i)

    # Perform train/test split
    domain_splits = {}
    for domain, indices in by_domain.items():
        # shuffle indices reproducibly
        perm = torch.randperm(len(indices)).tolist()
        indices = [indices[i] for i in perm]

        train_size = int(train_ratio * len(indices))

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        domain_splits[domain] = {
            "train": Subset(dst_all, train_indices),
            "test": Subset(dst_all, test_indices),
        }
    
    # Use config to determine which data is train data and construct auxiliary test sets
    train_domains = config['dataset']['train_domains']
    aux_test_domains = config['dataset']['aux_test_domains']

    dst_train = ConcatDataset([domain_splits[domain]['train'] for domain in train_domains])
    dst_test = ConcatDataset([domain_splits[domain]['test'] for domain in train_domains])
    
    aux_test_dsts = []
    for domain in aux_test_domains:
        aux_test_dsts.append(domain_splits[domain]['test'])
        # If the "train" partition of the domain is not being used in training, use it for auxiliary testing
        if domain not in train_domains:
            aux_test_dsts.append(domain_splits[domain]['train'])
    aux_test_dsts = ConcatDataset(aux_test_dsts)

    return dst_train, dst_test, aux_test_dsts


def PACS(config, logger, keep_classes: set = None):
    orig_size = 227
    im_size = (orig_size, orig_size) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    
    if keep_classes is None:
        num_classes = 7
        keep_classes = set(range(len(pacs_classes))) # Set of numbers 0 through 6
    else:
        num_classes = len(keep_classes)

    mean = [0.7615, 0.7420, 0.7134]
    std = [0.3081, 0.3173, 0.3455]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == orig_size else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == orig_size else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_all = load_dataset("flwrlabs/pacs", split="train") # "train" is the only available split in this dataset 

    dst_train, dst_test, aux_test_dsts = _train_test_split(config, dst_all, keep_classes)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='PACS',
        dst_train=dst_train,
        dst_test=dst_test,
        auxiliary_test_dsts=aux_test_dsts,
        num_classes=num_classes,
        classes=pacs_classes,
        templates=pacs_templates,
        transform=transform,
        test_transform=test_transform,
    )

# def PACS3(config, logger):
#     # Keep 3 PACS classes: dog(0), guitar(3), house(5)
#     keep = [0, 3, 5]
#     pacs3_classes = pacs_classes[keep]
#     assert pacs3_classes == ['dog', 'guitar', 'house']

#     im_size = (227, 227) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
#     num_classes = 3
#     # TODO:
#     # mean = [0.4914, 0.4822, 0.4465]
#     # std = [0.2470, 0.2435, 0.2616]

#     transform = transforms.Compose(
#         [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)]
#         ) if im_size[0] == 32 else transforms.Compose(
#         [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)]
#         )

#     test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
#         [transforms.Resize(im_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)]
#         )
    
#     dst_train = datasets.PACS7(
#         config['dataset']['root'], train=True, download=True, transform= transform
#     )
    
    
#     dst_test = datasets.PACS7(config['dataset']['root'], train=False, download=True, transform = test_transform)

#     # Filter train split in-place so we preserve CIFAR dataset fields (.data/.targets)
#     train_targets = np.array(dst_train.targets, dtype=np.int64)
#     train_mask = np.isin(train_targets, keep)
#     dst_train.data = dst_train.data[train_mask]
#     # Remap labels to contiguous [0, 1, 2] for safety
#     label_map = {old: new for new, old in enumerate(keep)}
#     dst_train.targets = [label_map[int(t)] for t in train_targets[train_mask]]

#     # Filter test split in-place with the same mapping
#     test_targets = np.array(dst_test.targets, dtype=np.int64)
#     test_mask = np.isin(test_targets, keep)
#     dst_test.data = dst_test.data[test_mask]
#     dst_test.targets = [label_map[int(t)] for t in test_targets[test_mask]]

#     return _build_dataset_info(
#         config=config,
#         logger=logger,
#         dataset_name='CIFAR3',
#         dst_train=dst_train,
#         dst_test=dst_test,
#         num_classes=num_classes,
#         classes=pacs3_classes,
#         templates=pacs_templates,
#     )