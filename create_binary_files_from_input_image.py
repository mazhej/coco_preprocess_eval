import torch
from pathlib import Path
import os
import json
# 
import utils
from models import load_model
from dataset import get_coco_dataloader

def dg_create_binary_files(args):

    # Loading val dataset
    data_loader_test = get_coco_dataloader(args)
    # Loading model
    model = load_model(args)
    # patch_fastrcnn(model)
    device = torch.device(args.device)
    model.to(device)
    
    if (args.model == 'yolo'):
        if (args.dataset == 'coco2014'):
            preprocess_and_save_bin_yolo_2014(model, data_loader_test, device=device)
        else:
            preprocess_and_save_bin_yolo_2017(model, data_loader_test, device=device)
    else: # non Yolo detection included in Torchvision
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
def preprocess_and_save_bin_yolo_2014(model, data_loader, device):
    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    shape_dict = {}
    for imgs, targets, paths, shapes in metric_logger.log_every(data_loader, 10, header):

        image_id = int(Path(paths[0]).stem.split('_')[-1])

        filePath = os.path.join( args.output_dir, str(image_id) + '.bin')
        transformed_np_img = imgs[0].cpu().numpy()
        transformed_np_img.tofile(filePath) 
        shape_dict[str(image_id)] = [transformed_np_img.shape[1:], shapes]

    # gather the stats from all processes
    jsonPath = os.path.join( args.output_dir, 'images_shape.json')
    with open(jsonPath, 'w') as fp:
        json.dump(shape_dict, fp)

@torch.no_grad()
def preprocess_and_save_bin_yolo_2017(model, data_loader, device):
    #  Not Complete !
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    jdict = []
    for imgs, targets, paths, shapes in metric_logger.log_every(data_loader, 10, header):

        image_id = int(Path(paths[si]).stem.split('_')[-1])


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/media/mehrdad/3dd9d6bb-3b3b-426f-b47f-a87ad0ad8559/ml-data/COCO/2017', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--bin',default="/home/maziar/WA/Git/coco_preprocess_eval/images_eval_bin")
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--output-dir', default='./coco2014_eval_bin', help='path where to save')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--weights', default='yolo_weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--cfg', type=str, default='yolo_cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument("--rect", help="keep rectangular shapes", action="store_true")

    parser.add_argument("--evaluate", dest="test_only", help="Only test the model", action="store_true")    
    parser.add_argument("--bin-evaluate", dest="bin_evaluate", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true")

    args = parser.parse_args()
    # force batch_size = 1
    args.batch_size = 1
    # 
    if args.output_dir:
        utils.mkdir(args.output_dir)

    dg_create_binary_files(args)
