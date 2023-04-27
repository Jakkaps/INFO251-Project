import torch
import numpy as np
from torchvision.transforms import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image as PilImage

class MaskGenerator(object):
    """
    Wrapper around Meta AI's segment anything model to generate masks for images.
    """

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)

    
    def __call__(self, img: PilImage) -> torch.Tensor:
        image_as_np = np.array(img).astype(np.uint8)
        masks = self.mask_generator.generate(image_as_np)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        masks_t = torch.cat([torch.from_numpy(m['segmentation']).float().unsqueeze(0) for m in masks], dim=0)
        return masks_t