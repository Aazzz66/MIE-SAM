from typing import Union, List
import math
import torch.nn
import torch.nn.functional as F

from torch import nn
from torch.nn import Module


def feature_fusion(joint_feats: torch.Tensor, joint_feat_masks: torch.Tensor) -> torch.Tensor:
    joint_feat_masks = F.softmax(joint_feat_masks, dim=-1)
    # (B, N_patch, N_filter, feat_num) -> (N_filter, B, N_patch, feat_num) -> (N_filter, B, N_patch, 1, feat_num)
    joint_feat_masks = joint_feat_masks.permute(2, 0, 1, 3).unsqueeze(-2)

    # (B, N_patch, C, feat_num) -> (1, B, N_patch, C, feat_num)
    joint_feats = joint_feats.unsqueeze(0)
    joint_feats = joint_feats * joint_feat_masks
    joint_feats = torch.sum(joint_feats, dim=-1)
    joint_feats = torch.mean(joint_feats, dim=0)
    # print("joint_feats:", joint_feats.shape)
    return joint_feats


def feature_fusion_by_rgb_x(
        rgb_feats: torch.Tensor, x_feats: torch.Tensor, rgb_feat_masks: torch.Tensor, x_feat_masks: torch.Tensor):
    joint_feat_masks = torch.cat(
        (rgb_feat_masks.unsqueeze(-1), x_feat_masks.unsqueeze(-1)), dim=-1
    )
    # (B, N_patch, C, 1) * 2 -> (B, N_patch, C, 2)
    joint_feats = torch.cat(
        (rgb_feats.unsqueeze(-1), x_feats.unsqueeze(-1)), dim=-1
    )
    return feature_fusion(joint_feats, joint_feat_masks)


class DynamicFusionModule(nn.Module):
    def __init__(
            self,
            in_channels_per_feat: int = 256,
            filter_num: int = 1,
            feat_num: int = 2,
            intermediate_channels: Union[int, List[int], None] = None,
            filter_type: str = 'conv2d'
    ):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = []
        elif isinstance(intermediate_channels, int):
            intermediate_channels = [intermediate_channels]
        self.filter_type = filter_type
        self.feat_num = feat_num
        in_channels = in_channels_per_feat * feat_num

        if filter_type == 'linear':
            filter_bank, layer_in_features = [], in_channels
            for layer_out_features in intermediate_channels:
                filter_bank.extend(
                    [nn.Linear(in_features=layer_in_features, out_features=layer_out_features), nn.GELU()]
                )
                layer_in_features = layer_out_features
            filter_bank.append(nn.Linear(in_features=layer_in_features, out_features=filter_num * feat_num))
            self.filter_bank = nn.Sequential(*filter_bank)

        elif filter_type == 'conv2d':
            self.filter_bank = Conv2dFilterBank(
                in_channels=in_channels,
                filter_num=filter_num * feat_num,
                intermediate_channels=intermediate_channels
            )

        else:
            raise NotImplementedError(f"Filter type {filter_type} not implemented!")

    def forward(self, feat_list: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        for f in feat_list:
            assert f.shape == feat_list[0].shape, "All features must have the same shape"

        joint_feats = torch.stack(feat_list, dim=-1)

        bs, c, h, w, feat_num = joint_feats.shape
        joint_feats = joint_feats.flatten(2, 3).permute(0, 2, 3, 1)
        joint_feats_flattened = joint_feats.reshape(bs, h * w, -1)
        joint_feat_masks = self.filter_bank(joint_feats_flattened)
        joint_feat_masks = joint_feat_masks.reshape(bs, h * w, -1, feat_num)

        joint_feats = joint_feats.reshape(bs, h * w, feat_num, -1).permute(0, 1, 3, 2)
        joint_feats = feature_fusion(joint_feats, joint_feat_masks)
        joint_feats = joint_feats.reshape(bs, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)

        filter_num = joint_feat_masks.size(-2)
        joint_feat_masks = joint_feat_masks.reshape(bs, h, w, filter_num, feat_num)

        return joint_feats, joint_feat_masks


class Conv2dFilterBank(Module):
    def __init__(self, in_channels: int = 256, filter_num: int = 1, kernel_size: int = 3,
                 intermediate_channels: Union[int, List[int], None] = None):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd number!"
        padding = kernel_size // 2

        conv_filter_bank, layer_in_channels = [], in_channels
        for layer_out_channels in intermediate_channels:
            conv_filter_bank.extend(
                [nn.Conv2d(
                    in_channels=layer_in_channels, out_channels=layer_out_channels,
                    kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(layer_out_channels),
                    nn.GELU(),
                ]
            )
            layer_in_channels = layer_out_channels
            # print('layer_out_channels',layer_out_channels)
        conv_filter_bank.append(
            nn.Conv2d(in_channels=layer_in_channels, out_channels=filter_num, kernel_size=kernel_size, padding=padding)
        )
        self.conv_filter_bank = nn.Sequential(*conv_filter_bank)
        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get the height and width for reshaping
        side_len = int(math.sqrt(x.shape[-2]))
        # (B, N_patch, C) -> (B, C, N_patch) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, side_len, side_len)
        # (B, C, H, W) -> (B, N_filter, H, W)
        x = self.conv_filter_bank(x)
        # (B, N_filter, H, W) -> (B, N_filter, N_patch) -> (B, N_patch, N_filter)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

