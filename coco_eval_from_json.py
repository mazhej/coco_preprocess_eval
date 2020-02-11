from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
from pathlib import Path

from yolo_utils.datasets import LoadImagesAndLabels
import json
import numpy as np

dataset = LoadImagesAndLabels('yolo_data/5k.txt', img_size=416, batch_size=32, rect=True)
imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]

imgIds=sorted(imgIds)

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
cocoGt = COCO(glob.glob('/media/mehrdad/3dd9d6bb-3b3b-426f-b47f-a87ad0ad8559/ml-data/COCO/2014/images/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
# cocoGt = COCO('/media/mehrdad/3dd9d6bb-3b3b-426f-b47f-a87ad0ad8559/ml-data/COCO/2017/annotations/instances_val2017.json')  # initialize COCO ground truth api

anns = json.load(open('Yolov3_d_f.json'))
cocoDt = cocoGt.loadRes('Yolov3_d_f.json')  # initialize COCO pred api


imgIds = [ann['image_id'] for ann in anns]
imgIds_unq = np.unique(imgIds)
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds_unq  # [:32]  # only evaluate these images
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)