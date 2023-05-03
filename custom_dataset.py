import os
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset, FGVCAircraft
from typing import Any, Callable, Optional, Tuple, Dict
from mask_generator import MaskGenerator


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
        augmentations: Dict[str, object] = {},
        transform: Optional[Callable] = None,
        split: str = "train",
        number_of_mask_channels: int = 0
    ) -> None:
        super(FGVCAircraftAugmented, self).__init__(
            root=root,
            transform=transform
        )

        self.number_of_mask_channels = number_of_mask_channels
        self.mask_generator = None
        if self.number_of_mask_channels > 0:
            self.mask_generator = MaskGenerator()
        mask_path = os.path.join(self.root, 'fgvc-aircraft-augmented', 'masks')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        
        # Needed to resize mask appropriately
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                self.resize_mask = transforms.Resize(t.size, antialias=True)
                break
        
        _ = FGVCAircraft(root=root, split=split, download=True)
        folder_name = 'fgvc-aircraft-augmented'
        data_dir = os.path.join(self.root, folder_name, split)
        already_exists = os.path.exists(data_dir)

        if not already_exists:
            os.makedirs(data_dir)
            os.makedirs(os.path.join(data_dir, 'images'))


        # Load classes to convert between labels and indeces
        orig_class_file = os.path.join(
            self.root, "fgvc-aircraft-2013b", "data", "variants.txt"
        )
        with open(orig_class_file, "r") as f:
            self.classes = [line.strip() for line in f]
        self.label_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # Load FGVC-Aircraft dataset and save augmented version to disk
        orig_labels_file = os.path.join(
            self.root, "fgvc-aircraft-2013b", "data", f"images_variant_{split}.txt"
        )

        orig_data_dir = os.path.join(self.root, "fgvc-aircraft-2013b", "data", "images")
        orig_image_names = []
        orig_labels = []

        with open(orig_labels_file, "r") as f:
          lines = f.readlines()[1:]
          for line in lines:
            image_name, label_name = line.strip().split(" ", 1)
            orig_image_names.append(f"{image_name}.jpg")
            orig_labels.append(self.label_to_idx[label_name])

        new_data_dir = os.path.join(self.root, folder_name, split, "images")
        new_image_names = []
        new_images_labels = []

        # Create augmented dataset
        for image_name in orig_image_names:

            image_file = os.path.join(orig_data_dir, image_name)
            image = Image.open(image_file)
            
            # Add original image to dataset
            new_image_names.append(image_name)
            new_images_labels.append(orig_labels[orig_image_names.index(image_name)])
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
        
        # Generate mask if it does not exist yet
        mask = None
        if self.number_of_mask_channels > 0:
            mask_path = os.path.join(self.root, 'fgvc-aircraft-augmented', 'masks', image_name.replace('.jpg', '.pt'))
            mask_not_generated = not os.path.exists(mask_path)
            if mask_not_generated:
                mask = self.mask_generator(image)
                torch.save(mask, mask_path)
            else:
                mask = torch.load(mask_path)

        if self.transform is not None:
            image = self.transform(image)

        # Add mask to image as channels
        if not mask is None: 
            if not self.resize_mask is None:
                mask = self.resize_mask(mask)
            
            image = torch.cat((image, mask[:self.number_of_mask_channels]), dim=0)

        return image, label

    def __len__(self) -> int:
        return len(self.image_names)
