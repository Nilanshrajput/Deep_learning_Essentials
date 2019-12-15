import json
import os
from collections import namedtuple
import zipfile

from PIL import Image


class CityscapeDataset(Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            
        split (string, optional): The image split to use, ``train``, ``train_extra`` or ``val``
       
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``  
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

      
    """

    # Based on https://github.com/mcordts/cityscapesScripts
  

    def __init__(self, root=None, split='train_extra', transforms=None):
        if root is not None:
          self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        else:
          self.images_dir = os.path.join('leftImg8bit', split)
        self.split = split
        self.images = []
        self.root = root
        self.transforms = transforms
  
        #valid_modes = ("train", "train_extra", "val")
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            images_city=[]
            for file_name in os.listdir(img_dir):
                
                images_city.append(os.path.join(img_dir, file_name))
            # re.split(\d+,input) splits by integer value
            images_city.sort( key= lambda text: int(re.split('(\d+)',text)[3]+re.split('(\d+)',text) [5] ))
            self.images+=images_city
        
    def paths_dir(self):
      return(self.images)

    def im_path(self):
      return self.images            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
             image 
        """

        image = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        return image


    def __len__(self):
        return len(self.images)
