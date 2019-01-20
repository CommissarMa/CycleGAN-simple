import numpy as np
import visdom
from .tensor2image import tensor2image
import torch


class Visualizer():
    def __init__(self,opt):
        self.name=opt.name
        self.display_id=opt.display_id#visdom展示的第一个窗口的id，其他窗口的id在此基础上不断+1
        self.win_size=opt.display_winsize#visdom和html都用到的窗体尺寸
        
        self.vis=visdom.Visdom(server=opt.display_server,port=opt.display_port, env=opt.display_env, raise_exceptions=True)
            
    def display_images(self,visuals):
        """
        visuals：OrderDict类型，键为图像的名字如real_A，值为对应的numpy图像[channel，height，width]
        """
        title = self.name
        i=0
        for label, image in visuals.items():
            people_count=torch.sum(image)
            image_numpy=tensor2image(image)
            self.vis.image(image_numpy, win=self.display_id + 1+i, 
                           opts=dict(title=label+"  "+str(people_count),caption='需要把图片移开才能看见'))
            i+=1
                