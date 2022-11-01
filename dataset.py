import os
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

format = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class MyDataset():
    def __init__(self, src_dir):

        self.root = src_dir
        self.path_arr = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(format):
                    path = os.path.join(root,file)

                    self.path_arr.append(path)

    def __getitem__(self, index):
        data =  self.path_arr[index]
        return data

    def __len__(self):
        return len(self.path_arr)
        