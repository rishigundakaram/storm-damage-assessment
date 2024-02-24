from typing import Callable, Dict, List, Optional, Union


import torch
import torchvision
# import feature_pyramid_network
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import FasterRCNN

  
def get_model(encoder: str = 'resnet34', feature_extractor: str = 'fpn', detector: str ='rpn'):
    if encoder == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=True)
    elif encoder == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=True)
    elif encoder == 'resnet34':
        encoder = torchvision.models.resnet34(pretrained=True)
    else:
        raise ValueError(f"Encoder {encoder} not supported")

    if feature_extractor == 'fpn':
        feature_extractor = fpn_extractor(encoder)
        pass
    else: 
        raise ValueError(f"{feature_extractor = } not supported")
    
    backbone = SiameseBackbone(encoder, feature_extractor)
    if detector == 'rpn':
        model = FasterRCNN(backbone, num_classes=5)

    return model 

def fpn_extractor(
    encoder, 
    norm_layer = misc_nn_ops.FrozenBatchNorm2d,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> FeaturePyramidNetwork:
    # need in_channels_list, out_channels, extra_blocks, norm_layer

    returned_layers = [1, 2, 3, 4]
    in_channels_stage2 = encoder.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    print(in_channels_list)
    return FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer= norm_layer,
        )

class SiameseBackbone(torch.nn.Module): 
    def __init__(self, encoder, feature_extractor) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.out_channels = 256

    def forward(self, pre_image, post_image):
        pre_features = self.encoder(pre_image)
        post_features = self.encoder(post_image)
        features = self.feature_extractor(pre_features - post_features)
        return self.detector(features)
    
