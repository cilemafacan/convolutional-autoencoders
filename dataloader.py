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

    def get_concat_w(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    def iterdata(self):
        if self.data == None:
            image = Image.open(self.dataset[self.last_idx])
            self.height =image.height
            self.width = image.width

            image_w_s = image.crop((0, 0, self.size - 1, self.height))
            dst = self.get_concat_w(image, image_w_s)

            image_h_s = dst.crop((0, 0, dst.width, self.size - 1))
            out = self.get_concat_h(dst, image_h_s)
            #out.save("padding.png")
            if self.transform is not None:
                self.data = self.transform(out).to("cuda")

            self.last_idx += 1

        return self.data, self.width, self.height

    def __getitem__(self, idx):

        img, width, height = self.iterdata()
        
        self.patch = []
        if self.last_y >=  height:
            #print("Last x:", self.last_x, "Last y:", self.last_y)

            self.last_x = 0
            self.last_y = 0
            if self.last_idx == len(self.dataset):
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
        #random.shuffle(self.patch)
        stack = torch.stack(self.patch)
        return stack




                            
                                              

                                
