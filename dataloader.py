import numpy as np
import torch
import random

from PIL import Image


Image.MAX_IMAGE_PIXELS = 933120000

class PatchDataLoader():
    def __init__(self, dataset, transform=None, kernel_size=256, stride=1, batch_size=1):
        
        self.dataset = dataset
        self.transform = transform
        self.size    = kernel_size
        self.stride  = stride
        self.batch_size = batch_size
        self.last_x = 0
        self.last_y = 0
        self.width = 0
        self.height = 0
        self.last_idx = 0
        self.data = None

    def iterdata(self):
        if self.data == None:
            #print(self.dataset[self.last_idx])
            image = Image.open(self.dataset[self.last_idx]).convert("RGB")
            self.height =image.height
            self.width = image.width

            imageArray = np.asarray(image)
            
            valueX = imageArray[0:imageArray.shape[0], 0:self.size - 1, :]
            imageArray1 = np.hstack((imageArray, valueX))
            
            valueY = imageArray1[0:self.size - 1, 0:imageArray1.shape[1] , :]
            imageArray2 = np.vstack((imageArray1, valueY))
            
            imageTest = Image.fromarray(imageArray2)
            imageTest.save(f"padding_{self.last_idx}.png")
            
            if self.transform is not None:
                self.data = self.transform(imageArray2).to("cuda")

            self.last_idx += 1

        return self.data, self.width, self.height

    def __getitem__(self, idx):

        img, width, height = self.iterdata()
        self.patch = []
        
        if self.last_y >=  height:
            self.last_x = 0
            self.last_y = 0
            self.data = None
            
            if self.last_idx == len(self.dataset):
                self.last_idx = 0
                return None
        
        for y in range(self.last_y,  height, self.stride):
            if len(self.patch) == self.batch_size:
                break

            for x in range(self.last_x , width, self.stride):
                data = img[:,y:self.size+y, x:self.size+x]
                self.patch.append(data)
                self.last_x = x + self.stride 

                if self.last_x >= width:
                    self.last_x = 0
                    self.last_y = y + self.stride
                
                if len(self.patch) == self.batch_size:
                    break
        random.shuffle(self.patch)
        stack = torch.stack(self.patch)
        return stack




                            
                                              

                                
