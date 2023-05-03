import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .navigator import Navigator
from .resnet import resnet50
from core.utils import generate_default_anchor_maps, hard_nms
from torchvision.models.resnet import ResNet50_Weights

class NTSModel(nn.Module):

  def __init__(self, top_n=4, cat_num=4, n_classes=100, image_height=224, image_width=224) -> None:
    super(NTSModel, self).__init__()
    
    self.top_n = top_n
    self.n_classes = n_classes
    self.image_height = image_height
    self.image_width = image_width
    self.cat_num = cat_num
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup feature extractor
    self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    self.resnet.fc = nn.Linear(512 * 4, self.n_classes)

    self.concat_net = nn.Linear(2048 * (self.cat_num + 1), self.n_classes)
    self.partcls_net = nn.Linear(512 * 4, self.n_classes)

    _, edge_anchors, _ = generate_default_anchor_maps(input_shape=(self.image_height, self.image_width))
    self.pad_side = self.image_width
    self.edge_anchors = (edge_anchors + self.pad_side).astype(int)

    # Setup navigator
    self.navigator = Navigator()

  def forward(self, x):
    batch_size, channels_in, img_h, img_w = x.size()

    resnet_out, resnet_feature_1, resnet_feature_2 = self.resnet(x)
    x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)

    scores = self.navigator(resnet_feature_1.detach())
    cdds = [
        np.concatenate((val.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(val)).reshape(-1, 1)), axis=1) for val in scores.data.cpu().numpy()
    ]
    top_n_cdds = np.array([hard_nms(x, topn=self.top_n, iou_thresh=0.25) for x in cdds])
    top_n_idxs = torch.from_numpy(top_n_cdds[:, :, -1].astype(np.int64)).to(self.device)
    top_n_proba = torch.gather(scores, dim=1, index=top_n_idxs)

    # Recreate images
    part_imgs = torch.zeros([batch_size, self.top_n, channels_in, img_h, img_w]).to(self.device)
    for i in range(batch_size):
      for j in range(self.top_n):
        [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(int)
        part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(img_h, img_w), mode='bilinear', align_corners=True)
    part_imgs = part_imgs.view(batch_size * self.top_n, channels_in, img_h, img_w)

    _, _, part_features = self.resnet(part_imgs.detach())
    part_feature = part_features.view(batch_size, self.top_n, -1)
    part_feature = part_feature[:, :self.cat_num, ...].contiguous()
    part_feature = part_feature.view(batch_size, -1)
    # concat_logits have the shape: B*200
    concat_out = torch.cat([part_feature, resnet_feature_2], dim=1)
    concat_logits = self.concat_net(concat_out)
    resnet_logits = resnet_out
    # part_logits have the shape: B*N*200
    part_logits = self.partcls_net(part_features).view(batch_size, self.top_n, -1)
    return [resnet_logits, concat_logits, part_logits, top_n_idxs, top_n_proba]