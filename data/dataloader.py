import json
import torch
import torchvision
import torchvision.transforms as transforms

from edflow import get_logger
from edflow.custom_logging import LogSingleton
from edflow.data.dataset import DatasetMixin
from edflow.util import edprint

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
import json
import yaml

class Dataset(DatasetMixin):
    def __init__(self, config, train=False):
        """Initialize the dataset to load training or validation images according to the config.yaml file. 
        
        :param DatasetMixin: This class inherits from this class to enable a good workflow through the framework edflow.  
        :param config: This config is loaded from the config.yaml file which specifies all neccesary hyperparameter for to desired operation which will be executed by the edflow framework.
        """
        # Create Logging for the Dataset
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        self.logger = get_logger("Dataset")
        self.config = config

        self.style_root = self.config["data"]["style_path"]
        self.content_root = self.config["data"]["content_path"]

        self.set_image_transform()
        self.set_random_state()
        self.indices = self.load_content_indices(train)
        self.art_list = self.load_art_list()


    def set_image_transform(self):
        """"Builds transformation according to config"""
        transformations = []

        #define crop function depending on config
        crop_size = self.config["data"]["transforms"]["crop"]["size"]

        if self.config["data"]["transforms"]["crop"]["type"] == "random":
            crop_function = transforms.RandomCrop(crop_size)
            self.logger.info("RandomCrop with size {} is applied.".format(crop_size))
        elif self.config["data"]["transforms"]["crop"]["type"] == "center":
            crop_function = transforms.CenterCrop(crop_size)
            self.logger.info("CenterCrop with size {} is applied.".format(crop_size))
        else:
            crop_function = transforms.CenterCrop(crop_size)
            self.logger.info("No valid crop type found. Input: {}, Possible inputs: random and center".format(self.config["data"]["transforms"]["crop"]["type"]))
            self.logger.info("CenterCrop with size {} is applied by default.".format(crop_size))
        transformations.append(crop_function)
        #resizing
        transformations.append(transforms.Resize(self.config["data"]["transforms"]["in_size"]))
        self.logger.info("Images are resized to {}.".format(self.config["data"]["transforms"]["in_size"]))
        #mirroring
        if self.config["data"]["transforms"]["mirror"]:
            transformations.append(transforms.RandomHorizontalFlip())
            self.logger.info("Images are randomly flipped")
        
        transformations = transforms.Compose([*transformations, transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.transform = transformations
        
    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])
        else:
            raise ValueError("Enter random_seed in config.")


    def load_content_indices(self, train):
        """Generates list of indices for either train or test subset."""
        names  = os.listdir(self.content_root)
        cut_ind = int(len(names) * self.config["data"]["validation_split"])
        if train:
            indices = np.arange(cut_ind)
        else:
            indices = np.arange(cut_ind, len(names))
        return indices
        
    def load_art_list(self):
        """Construct directory of from [[artist1, label1], [artist2, label2] ... ]"""
        artists = os.listdir(self.style_root)
        labels = np.eye(len(artists))

        art_list = []
        for a,l in zip(artists, labels):
            art_list.append([a, torch.Tensor(l)])
        
        return art_list
    
    def __len__(self):
        """This member function returns the length of the content dataset"""
        return len(self.indices)

    def get_example(self, idx):
        """Loads an example of the dataset namely 2 style images 2 content images and the respective artist and the label

        Args:
            idx (int): index of content image

        Returns:
            dict: example holding "index", "artist", "style1", "style2", "content1", "content2".
        """
        example = {}
        example["index"] = idx
        #style information
        art_ind = np.random.randint(len(self.art_list))
        example["artist"], example["label"] = self.art_list[art_ind]
        art_root = os.path.join(self.style_root, example["artist"])
        art_names = os.listdir(art_root)
        example["style1"] = self.transform(Image.open(os.path.join(art_root, art_names[np.random.randint(len(art_names))])))
        example["style2"] = self.transform(Image.open(os.path.join(art_root, art_names[np.random.randint(len(art_names))])))

        example["content1"] = self.transform(Image.open(os.path.join(self.content_root, os.listdir(self.content_root)[idx])))
        example["content2"] = self.transform(Image.open(os.path.join(self.content_root, os.listdir(self.content_root)[self.indices[np.random.randint(len(self.indices))]] )))
        return example


class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)

def test():  
    def get_config(config_path):
        with open(config_path) as file:
            config = yaml.full_load(file)
        return config

    def unix_path(path):
        return path.replace("\\", "/")

    config = get_config(unix_path(r"C:\Users\user\Desktop\Zeug\Style transfer\style_transfer\configs\test_config.yaml"))

    d = DatasetTrain(config)

    img = d[0]["style1"].numpy().transpose((1,2,0))
    plt.imshow((img+1)/2)
    plt.show()

    img = d[0]["style2"].numpy().transpose((1,2,0))
    plt.imshow((img+1)/2)
    plt.show()

    img = d[0]["content1"].numpy().transpose((1,2,0))
    plt.imshow((img+1)/2)
    plt.show()

    img = d[0]["content2"].numpy().transpose((1,2,0))
    plt.imshow((img+1)/2)
    plt.show()