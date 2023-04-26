import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset, FGVCAircraft
from typing import Any, Callable, Optional, Tuple, Dict


class FGVCAircraftAugmented(VisionDataset):
    """
    A custom dataset class for the FGVC-Aircraft dataset where new images are created by applying augmentations.

    Attributes:
    root: Root dir to save dataset to.
    augmentations: Augmentation to create new images. List of name:torchvision.transforms.
    tranform: Transform to apply to all images. Torch transform.
    split: Split to use. One of 'train', 'val', 'test'.
    """

    def __init__(
        self,
        root: str,
        augmentations: Dict[str, object] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
    ) -> None:
        super(FGVCAircraftAugmented, self).__init__(
            root=root, transform=transform
        )

        _ = FGVCAircraft(root=root, download=True)
        folder_name = 'fgvc-aircraft-augmented'
        data_dir = os.path.join(self.root, folder_name, split)
        already_exists = os.path.exists(data_dir)

        if not already_exists:
            os.makedirs(data_dir)
            os.makedirs(os.path.join(data_dir, 'images'))

        # Load FGVC-Aircraft dataset and save augmented version to disk
        orig_annotations_file = os.path.join(
            self.root, "fgvc-aircraft-2013b", "data", f"images_variant_{split}.txt"
        )
        with open(orig_annotations_file, "r") as f:
            lines = f.readlines()[1:]

        orig_image_names = []
        orig_labels = []

        for line in lines:
            fields = line.strip().split(" ")
            orig_image_names.append(fields[0] + '.jpg')
            orig_labels.append(fields[1])

        orig_data_dir = os.path.join(self.root, "fgvc-aircraft-2013b", "data", "images")
        
        new_data_dir = os.path.join(self.root, folder_name, split, "images")
        new_image_names = []
        new_images_labels = []

        # Create augmented dataset
        for image_name in orig_image_names:
            image_file = os.path.join(orig_data_dir, image_name)
            image = Image.open(image_file)
            
            # Add original image to dataset
            new_image_file = os.path.join(new_data_dir, image_name)
            if not already_exists:
                image.save(new_image_file)

            # Apply augmentations to image and save to disk
            for name, augmentation in augmentations.items():
                new_image_file = new_image_file.replace('.jpg', f'_{name}.jpg')
                augmentation_exists = os.path.exists(new_image_file)
                if not augmentation_exists:
                    augmented_image = augmentation(image)
                    augmented_image.save(new_image_file)
                new_image_names.append(f'{image_name}'.replace('.jpg', f'_{name}.jpg'))
                new_images_labels.append(orig_labels[orig_image_names.index(image_name)])
        
        self.image_labels = new_images_labels
        self.image_names = new_image_names
        self.data_dir = new_data_dir

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_name = self.image_names[index]
        label = self.image_labels[index]

        image_file = os.path.join(self.data_dir, image_name)

        with open(image_file, "rb") as f:
            image = Image.open(f).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.image_names)
