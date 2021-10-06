import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
# from vit_pytorch import ViT
import pdb
import timm


def get_my_model(model_name, num_classes, checkpoint_path=""):
    model_type = None



    # fot ai_connect
    if model_name == "tf_efficientnet_l2_ns_475":
        model_ft = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True, num_classes=num_classes)

    if model_name == "swin_large_patch4_window12_384":
        model_ft = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes)

    if model_name == "vit_large_patch16_384":
        model_ft = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=num_classes)

    if model_name == "cait_m48_448":
        model_ft = timm.create_model('cait_m48_448', pretrained=True, num_classes=num_classes)
        
    if model_name == "tf_efficientnetv2_l_in21ft1k":
        model_ft = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True, num_classes=num_classes)

    if model_name == "tf_efficientnetv2_m_in21ft1k":
        model_ft = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True, num_classes=num_classes)
    
    


    ########
    if model_name == "vgg16":
        model_ft = models.vgg16(pretrained=True)
        model_type = "vgg"

    if model_name == "vgg16_f":
        model_ft = models.vgg16(pretrained=False)
        model_type = "vgg"

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet18_mp":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1

    if model_name == "resnet18_f":
        model_ft = models.resnet18(pretrained=False)
        model_type = "res"

    if model_name == "resnet34":
        model_ft = models.resnet34(pretrained=True)
        model_type = "res"

    if model_name == "resnet34_f":
        model_ft = models.resnet34(pretrained=False)
        model_type = "res"

    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        model_type = "res"

    if model_name == "resnet50_f":
        model_ft = models.resnet50(pretrained=False)
        model_type = "res"

    if model_name == "resnet101":
        model_ft = models.resnet101(pretrained=True)
        model_type = "res"

    if model_name == "resnet101_f":
        model_ft = models.resnet101(pretrained=False)
        model_type = "res"

    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet152_mp":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"
        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1

    if model_name == "resnet152_f":
        model_ft = models.resnet152(pretrained=False)
        model_type = "res"

    if model_name == "wide_resnet50_2":
        model_ft = models.wide_resnet50_2(pretrained=True)
        model_type = "res"

    if model_name == "wide_resnet50_2_f":
        model_ft = models.wide_resnet50_2(pretrained=False)
        model_type = "res"

    if model_name == "wide_resnet101_2":
        model_ft = models.wide_resnet101_2(pretrained=True)
        model_type = "res"

    if model_name == "wide_resnet101_2_f":
        model_ft = models.wide_resnet101_2(pretrained=False)
        model_type = "res"

    if model_name == "resnext50_32x4d":
        model_ft = models.resnext50_32x4d(pretrained=True)
        model_type = "res"

    if model_name == "resnext50_32x4d_f":
        model_ft = models.resnext50_32x4d(pretrained=False)
        model_type = "res"

    if model_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=True)
        model_type = "res"

    if model_name == "resnext101_32x8d_f":
        model_ft = models.resnext101_32x8d(pretrained=False)
        model_type = "res"

    if model_name == "eff_b0":
        model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        model_type = "eff"

    if model_name == "eff_b2":
        model_ft = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
        model_type = "eff"

    if model_name == "eff_b4":
        model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        model_type = "eff"

    if model_name == "eff_b6":
        model_ft = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
        model_type = "eff"

    if model_name == "eff_b7":
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        model_type = "eff"

    if model_name == "densenet121":
        model_ft = models.densenet121(pretrained=True)
        model_type = "den"

    if model_name == "densenet121_f":
        model_ft = models.densenet121(pretrained=False)
        model_type = "den"

    if model_name == "densenet201_f":
        model_ft = models.densenet201(pretrained=False)
        model_type = "den"

    if model_name == "mnasnet1_0":
        model_ft = models.mnasnet1_0(pretrained=True)
        model_type = "mnas"






    # timm

    if model_name == "timm_resnet50":
        model_ft = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

    if model_name == "timm_resnet50_f":
        model_ft = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)

    if model_name == "timm_wide_resnet50_2":
        model_ft = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=num_classes)

    if model_name == "timm_wide_resnet50_2_f":
        model_ft = timm.create_model('wide_resnet50_2', pretrained=False, num_classes=num_classes)

    if model_name == "timm_resnext50_32x4d":
        model_ft = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=num_classes)

    if model_name == "timm_resnext50_32x4d_f":
        model_ft = timm.create_model('resnext50_32x4d', pretrained=False, num_classes=num_classes)

    if model_name == "timm_resnext101_32x8d":
        model_ft = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=num_classes)

    if model_name == "timm_eff_b0":
        model_ft = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    if model_name == "timm_tf_efficientnetv2_s":
        model_ft = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=num_classes)

    if model_name == "timm_tf_efficientnetv2_m":
        model_ft = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=num_classes, checkpoint_path=checkpoint_path)

    if model_name == "timm_tf_efficientnetv2_l":
        model_ft = timm.create_model('tf_efficientnetv2_l', pretrained=True, num_classes=num_classes)

    if model_name == "timm_tf_efficientnetv2_l_f":
        model_ft = timm.create_model('tf_efficientnetv2_l', pretrained=False, num_classes=num_classes)

    if model_name == "timm_vit_small_patch16_224":
        model_ft = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_vit_base_patch16_224":
        model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_vit_large_patch16_224":
        model_ft = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_vit_large_patch16_224_f":
        model_ft = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes)

    if model_name == "timm_vit_base_patch16_224_in21k":
        model_ft = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes)

    if model_name == "timm_vit_base_patch16_224_miil":
        model_ft = timm.create_model('vit_base_patch16_224_miil', pretrained=True, num_classes=num_classes)

    if model_name == "timm_mixer_b16_224":
        model_ft = timm.create_model('mixer_b16_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_mixer_l16_224":
        model_ft = timm.create_model('mixer_l16_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_swin_small_patch4_window7_224":
        model_ft = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_swin_base_patch4_window7_224":
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_swin_large_patch4_window7_224":
        model_ft = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=num_classes)

    if model_name == "timm_swin_large_patch4_window7_224_f":
        model_ft = timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=num_classes)




    # last fc configuration
    if model_type == "res":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == "vgg":
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == "den":
        num_ftrs = model_ft.classifier.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)


    return model_ft
