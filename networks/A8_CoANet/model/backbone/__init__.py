from networks.A8_CoANet.model.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    else:
        raise NotImplementedError
