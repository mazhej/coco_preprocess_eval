from yolo_models import *
import torchvision.models.detection as detection

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
    else:
        model = detection.__dict__[args.model](num_classes=91, pretrained=args.pretrained)
    
    return model