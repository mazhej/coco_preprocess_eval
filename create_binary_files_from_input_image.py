import torch
import torchvision.models.detection as detection
import io
import math
import sys
import time
import torch
import json
import os
import numpy as np
import torchvision.models.detection.mask_rcnn
# 
import utils
import transforms as T
from coco_utils import get_coco, get_coco_kp
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from coco_utils import get_coco_api_from_dataset
# 
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

from torchvision.models.detection.transform import GeneralizedRCNNTransform

class IdentityTransform(GeneralizedRCNNTransform):
   
    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            
            image = image
            image, target = image, target
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets

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
    
    if args.bin_evaluate:
        evaluate_bin(model, data_loader_test, device=device, bin_folder = args.bin)
    elif args.test_only:
        evaluate(model, data_loader_test, device=device)
    else:
        preprocess_and_save_bin(model, data_loader_test, device=device)


@torch.no_grad()
def preprocess_and_save_bin(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    shape_dict = {}
    for image, targets in data_loader:
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        transformed_image = model.transform(image)

        img_id = targets[0]['image_id'].cpu().numpy()[0]
        filePath = os.path.join( args.output_dir, str(img_id) + '.bin')
        transformed_np_img = transformed_image[0].tensors.cpu().numpy()
        transformed_np_img.tofile(filePath) 
        shape_dict[str(img_id)] = [[transformed_np_img.shape ],[ transformed_image[0].image_sizes[0][:]]]
    



    # gather the stats from all processes
    jsonPath = os.path.join( args.output_dir, 'images_shape.json')
    with open(jsonPath, 'w') as fp:
        json.dump(shape_dict, fp)

    torch.set_num_threads(n_threads)

@torch.no_grad()
def evaluate_bin(model, data_loader, device, bin_folder ):

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    bin_folder = bin_folder
    jsonPath = os.path.join( args.output_dir, 'images_shape.json')
    with open(jsonPath) as json_file:
        shape_dict = json.load(json_file)
    #  
    model.transform = IdentityTransform(model.transform.min_size, model.transform.max_size, model.transform.image_mean, model.transform.image_std)
    model.eval()
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        original_image_sizes = [img.shape[-2:] for img in image]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        img_id = targets[0]['image_id'].cpu().numpy()[0]
        path = os.path.join(bin_folder, str(img_id) +'.bin')
        f = open(path, 'rb')
        transformed_img = np.fromfile(f, np.float32)
        transformed_img = np.reshape(transformed_img, shape_dict[str(img_id)][0][0]) 
        
        image_sizes_not_devisible = np.asarray(shape_dict[str(img_id)][1][0])
        image_sizes_not_devisible=torch.from_numpy(image_sizes_not_devisible)

        transformed_img_T = torch.from_numpy(transformed_img)
        transformed_img_T = transformed_img_T.to(device)
       
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(transformed_img_T)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs = model.transform.postprocess(outputs, [image_sizes_not_devisible], original_image_sizes)

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--bin',default="/home/maziar/WA/Git/coco_preprocess_eval/images_eval_bin")
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--output-dir', default='./images_eval_bin', help='path where to save')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument(
        "--evaluate",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--bin-evaluate",
        dest="bin_evaluate",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    dg_main(args)
