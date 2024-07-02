import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import FCOS, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50, resnet101
from functools import partial 
import torchvision
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def fcos_model(num_classes):
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained = True)
    
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    

    return model

def fasterrcnn_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


def retinanet_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained = True)
    
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    

    return model

