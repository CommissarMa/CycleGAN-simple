import time
from options.train_options import TrainOptions
from data.unaligned_dataset import UnalignedDataset
from torch.utils.data import DataLoader
from util.visualizer import Visualizer
import numpy as np
from util.tensor2image import tensor2image
from models.cycle_gan_model import CycleGANModel
from collections import OrderedDict


if __name__=="__main__":
    start_time=time.asctime(time.localtime(time.time()))
    print("开始时间:",start_time)
    
    #设置所有参数
    opt=TrainOptions().get_options()#顺便会在控制台输出所有参数信息
    
    #创建数据集
    dataset=UnalignedDataset(opt)
    dataloader=DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=opt.num_threads)
    dataset_size=len(dataloader)
    print("数据集大小：", dataset_size, "张")
    
    #创建模型
    model=CycleGANModel(opt)
    model.setup(opt)
    
    #创建可视化
    visualizer=Visualizer(opt)
    
    #开始训练
    total_steps=0#总共训练了多少张图像
    for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
        epoch_start_time=time.time()#计算一个epoch花费的时间
        epoch_iter=0#当前epoch中训练了多少张图像
        
        for i,data in enumerate(dataloader):
            
            #训练一次
            model.set_input(data)
            model.optimize_parameters()
            
            total_steps+=1
            if total_steps % 10==0:
                visualizer.display_images(model.get_current_visuals())

#            break
        break
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    