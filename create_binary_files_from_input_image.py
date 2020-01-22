import torch
import time
import json
import os
import numpy as np
# 
import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import KeypointRCNN
# 
from models import load_model
from dataset import get_coco_dataloader
from yolo_utils.utils import coco80_to_coco91_class, non_max_suppression, clip_coords, scale_coords, xyxy2xywh, floatn
from pathlib import Path

coco91class = coco80_to_coco91_class()
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

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def dg_main(args):

    # Loading val dataset
    data_loader_test = get_coco_dataloader(args)
    # Loading model
    model = load_model(args)
    # patch_fastrcnn(model)
    device = torch.device(args.device)
    model.to(device)
    
    if args.bin_evaluate:
        evaluate_bin(model, data_loader_test, device=device, bin_folder = args.bin)
    elif args.test_only:
        if (args.model == 'yolo'):
            if (args.dataset == 'coco2014'):
                evaluate_yolo_2014(model, data_loader_test, device=device)
            else:
                evaluate_yolo_2017(model, data_loader_test, device=device)
        else: # non Yolo detection included in Torchvision
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

@torch.no_grad()
def evaluate_yolo_2017(model, data_loader, device):
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
    transform = GeneralizedRCNNTransform(416, 416, [0, 0, 0], [1, 1, 1])
    transform.eval()
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        original_image_sizes = [img.shape[-2:] for img in image]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        transformed_img = transform(image)
        transformed_shape = transformed_img[0].tensors.shape[-2:]
        inf_out, _ = model(transformed_img[0].tensors)
        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)

        # Statistics per image
        predictions = []
        for si, pred in enumerate(output):
            prediction = {  'boxes': [],
                            'labels': [],
                            'scores': []
                            }
            if pred is None:
                continue
            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, transformed_shape)
            # Append to pycocotools JSON dictionary
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            image_id = int(targets[si]['image_id'])
            box = pred[:, :4].clone()  # xyxy
            # scale_coords(transformed_shape, box, shapes[si][0], shapes[si][1])  # to original shape
            # box = xyxy2xywh(box)  # xywh
            # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for di, d in enumerate(pred):
                box_T = [floatn(x, 3) for x in box[di]]
                label = coco91class[int(d[5])]
                score = floatn(d[4], 5)
                prediction['boxes'].append( box_T )
                prediction['labels'].append( label )
                prediction['scores'].append( score )
            prediction['boxes'] = torch.tensor(prediction['boxes'])
            prediction['labels'] = torch.tensor(prediction['labels'])
            prediction['scores'] = torch.tensor(prediction['scores'])
            predictions.append( prediction )

            
        outputs = transform.postprocess(predictions, transformed_img[0].image_sizes, original_image_sizes)


        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions]
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

@torch.no_grad()
def evaluate_yolo_2014(model, data_loader, device):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    jdict = []
    for imgs, targets, paths, shapes in metric_logger.log_every(data_loader, 10, header):

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width
        
        torch.cuda.synchronize()
        model_time = time.time()
        inf_out, _ = model(imgs)  # inference and training outputs
        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
        model_time = time.time() - model_time
        # add eval to json
        evaluator_time = time.time()
        # Statistics per image
        for si, pred in enumerate(output):
            if pred is None:
                continue
            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            # Append to pycocotools JSON dictionary
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            image_id = int(Path(paths[si]).stem.split('_')[-1])
            box = pred[:, :4].clone()  # xyxy
            scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
            box = xyxy2xywh(box)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for di, d in enumerate(pred):
                jdict.append({'image_id': image_id,
                                'category_id': coco91class[int(d[5])],
                                'bbox': [floatn(x, 3) for x in box[di]],
                                'score': floatn(d[4], 5)})

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(glob.glob('/media/mehrdad/3dd9d6bb-3b3b-426f-b47f-a87ad0ad8559/ml-data/COCO/2014/images/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(jdict)  # initialize COCO pred api

    imgIds = [int(Path(x).stem.split('_')[-1]) for x in data_loader.dataset.img_files]
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco2017', help='dataset')
    parser.add_argument('--bin',default="/home/maziar/WA/Git/coco_preprocess_eval/images_eval_bin")
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--output-dir', default='./images_eval_bin', help='path where to save')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--weights', default='yolo_weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--cfg', type=str, default='yolo_cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument("--rect", help="keep rectangular shapes", action="store_true")

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
