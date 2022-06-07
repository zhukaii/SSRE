from torchvision import models
from utils.model.my_model import my_model


from utils.model.backbone.resnet18_no1 import resnet18_cbam1


my_models = {
    'my': my_model,
}

backbones = {
    'resnet18': models.resnet18,
    'resnet18_no1': resnet18_cbam1,
}


def prepare_model(args):
    if args.model_name:
        model = my_models[args.model_name](backbone=backbones[args.backbone_name], pretrained=args.pretrained, args=args)
    else:
        model = None
    return model
