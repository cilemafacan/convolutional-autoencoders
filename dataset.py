import os
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

supportedExt = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

class MyDataset():
    def __init__(self, srcDir):

        self.root = srcDir
        self.pathArr = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(supportedExt):
                    path = os.path.join(root,file)
                    self.pathArr.append(path)

    def __getitem__(self, index):
        data =  self.pathArr[index]
        return data

    def __len__(self):
        return len(self.pathArr)
        