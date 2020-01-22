import torch
from torch.utils.data import DataLoader
from yolo_utils.datasets import LoadImagesAndLabels
import transforms as T
from coco_utils import get_coco, get_coco_kp

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_coco_dataloader(args):
    if (args.dataset == 'coco2014'):
        dataset_test = LoadImagesAndLabels('yolo_data/5k.txt', img_size=416, batch_size=args.batch_size, rect=args.rect)
        args.batch_size = min(args.batch_size, len(dataset_test))
        data_loader_test = DataLoader(dataset_test,
                                batch_size=args.batch_size,
                                num_workers=args.workers, #min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset_test.collate_fn)
    else:   # coco2017
        dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, 
                                    batch_size=args.batch_size,
                                    sampler=test_sampler, 
                                    num_workers=args.workers,
                                    collate_fn=collate_fn)
    return data_loader_test