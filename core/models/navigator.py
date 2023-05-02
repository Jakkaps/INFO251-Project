import torch
import torch.nn as nn
from typing import TypedDict

class ConvLayer(TypedDict):
  out_channels: int
  kernel_size: int
  stride: int
  padding: int

class NavigatorBlock(nn.Module):
  def __init__(self, in_channels, conv1_params: ConvLayer, conv2_params: ConvLayer) -> None:
      super(NavigatorBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_channels, **conv1_params)
      self.relu = nn.ReLU()
      self.conv2 = nn.Conv2d(conv1_params["out_channels"], **conv2_params)

  def forward(self, x):
    batch_size = x.size(0)
    x1 = self.conv1(x)
    x1 = self.relu(x1)
    x2 = self.conv2(x1)
    return x1, x2.view(batch_size, -1)


class Navigator(nn.Module):

  def __init__(self) -> None:
    super(Navigator, self).__init__()
    self.block1 = NavigatorBlock(
        2048, 
        conv1_params=ConvLayer(out_channels=128, kernel_size=3, stride=1, padding=1), 
        conv2_params=ConvLayer(out_channels=6, kernel_size=1, stride=1, padding=0)
    )
    self.block2 = NavigatorBlock(
        128, 
        conv1_params=ConvLayer(out_channels=128, kernel_size=3, stride=2, padding=1), 
        conv2_params=ConvLayer(out_channels=6, kernel_size=1, stride=1, padding=0)
    )
    self.block3 = NavigatorBlock(
        128, 
        conv1_params=ConvLayer(out_channels=128, kernel_size=3, stride=2, padding=1), 
        conv2_params=ConvLayer(out_channels=9, kernel_size=1, stride=1, padding=0)
    )

  def forward(self, x):
    _x1, x1 = self.block1(x)
    _x2, x2 = self.block2(_x1)
    _, x3 = self.block3(_x2)
    return torch.cat((x1, x2, x3), dim=1)