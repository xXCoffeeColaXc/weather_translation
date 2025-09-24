from typing import Callable, Union, Optional
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from seg.utils.ext_transform import ExtCompose


class ACDCDataset(Dataset):

    CityscapesClass = namedtuple(
        'CityscapesClass',
        ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

    # TODO: add weights to the class
    #     # Define the class weights (example values, can be adjusted)
    # weights = {
    #     0: 1.0,    # road
    #     1: 1.0,    # sidewalk
    #     2: 2.0,    # building
    #     3: 2.5,    # wall
    #     4: 2.0,    # fence
    #     5: 1.5,    # pole
    #     6: 3.0,    # traffic light
    #     7: 2.5,    # traffic sign
    #     8: 1.0,    # vegetation
    #     9: 1.2,    # terrain
    #     10: 1.0,   # sky
    #     11: 3.0,   # person
    #     12: 2.5,   # rider
    #     13: 1.0,   # car
    #     14: 1.5,   # truck
    #     15: 1.5,   # bus
    #     16: 3.0,   # train
    #     17: 2.5,   # motorcycle
    #     18: 2.0,   # bicycle
    # }

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    # Create a dictionary mapping class IDs to class names
    class_id_to_name = {c.train_id: c.name for c in classes}
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 weather: Union[str, list[str]] = 'all',
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 target_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        """
        Args:
            root_dir (str or Path): Root directory of the dataset (parent of 'gt' and 'rgb_anon' folders).
            split (str): One of 'train', 'val', 'test', etc.
            weather (str): One of 'fog', 'night', 'rain', 'snow', or 'all' to include all conditions.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on a label.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.weather = weather
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths = []
        self.label_paths = []

        if weather == 'all':
            weather_conditions = ['fog', 'night', 'rain', 'snow']
        #elif all(item in ['fog', 'night', 'rain', 'snow'] for item in weather):
        else:
            weather_conditions = weather

        for condition in weather_conditions:
            rgb_dir = self.root_dir / 'rgb_anon' / condition / split
            gt_dir = self.root_dir / 'gt' / condition / split

            # For each image in rgb_dir, find the corresponding label in gt_dir
            image_pattern = '**/*_rgb_anon.*'  # Matches files ending with '_rgb_anon.png' or '_rgb_anon.jpg'
            for image_path in rgb_dir.glob(image_pattern):
                # Construct relative path to match label
                relative_path = image_path.relative_to(rgb_dir)
                label_filename = image_path.name.replace('_rgb_anon', '_gt_labelIds')
                label_path = gt_dir / relative_path.parent / label_filename

                if label_path.exists():
                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)
                else:
                    print(f"Warning: Label not found for image {image_path}")

    @classmethod
    def encode_target(cls, target):
        target = np.array(target)
        return cls.id_to_train_id[target]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Apply transforms if any
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = torch.from_numpy(np.array(image)).float()
            label = torch.from_numpy(np.array(label)).long()

        label = self.encode_target(label)
        label = torch.from_numpy(np.array(label)).long()
        # if self.target_transform:
        #     label = self.target_transform(label)
        #     label = self.encode_target(label)
        #     label = torch.from_numpy(np.array(label)).long()
        # else:
        #     # Convert label to tensor
        #     label = torch.from_numpy(np.array(label)).long()

        return image, label


def get_dataloader(root_dir: str = 'data',
                   split: str = 'train',
                   weather: Union[str, list[str]] = 'all',
                   transform: ExtCompose = None,
                   batch_size: int = 4,
                   shuffle: bool = True,
                   num_workers: int = 4) -> DataLoader:
    # Create the dataset
    dataset = ACDCDataset(root_dir=root_dir, split=split, weather=weather, transform=transform)

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the dataset path.")

    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
