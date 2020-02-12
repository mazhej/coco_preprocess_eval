from yolo_models import *
import torchvision.models.detection as detection
from vision import *
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

voc_class_names = ['BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def load_model(args):
    device = torch.device(args.device)
    print("Loading model")
    if (args.model == 'yolo'):
        # Initialize model
        model = Darknet(args.cfg, img_size=416)
        # Load weights
        attempt_download(args.weights)
        if args.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, args.weights)
    elif (args.model== 'mobilenetv1_ssd'):
        model = create_mobilenetv1_ssd(len(voc_class_names), is_test=True)
    else:
        model = detection.__dict__[args.model](num_classes=91, pretrained=args.pretrained)
    
    return model