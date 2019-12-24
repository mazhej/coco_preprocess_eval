import torch
import utils
import transforms as T
from coco_utils import get_coco, get_coco_kp
import torchvision.models.detection as detection
from engine import train_one_epoch, evaluate
import io
import math
import sys
import time
import torch
import json
from coco_utils import get_coco, get_coco_kp
import os
import numpy as np
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
buffer = io.BytesIO()
annotation_file = "/home/maziar/WA/eval_res/distiller/examples/object_detection_compression/data/annotations/instances_val2017.json"
bin_folder = "/home/maziar/WA/eval_res/distiller/bins/"

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def dg_main(args):

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=0,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    # patch_fastrcnn(model)
    model.to(device)
    
    
    #evaluate(model, data_loader_test, device=device)
        
    
    

    ####
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
     #creating binary files from images after transformation
        outputs[0].tensors.cpu().numpy().tofile(str(targets[0]['image_id'].numpy()[0]) +'.bin') 
        my_dictionary[targets[0]['image_id'].numpy()[0]] = outputs[0].tensors.cpu().numpy().shape
        for key in my_dictionary.keys():
           if type(key) is not str:
                try:
                   my_dictionary[str(key)] = my_dictionary[key]
                except:
                    try:
                        my_dictionary[repr(key)] = my_dictionary[key]
                    except:
                        pass
                del my_dictionary[key]
    with open('data.json', 'w') as fp:
        json.dump(my_dictionary, fp)

    torch.cuda.synchronize()
    model_time = time.time()
    outputs = model.transform(data)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--evaluate",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # add_distiller_compression_args(parser)

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    dg_main(args)
