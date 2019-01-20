import numpy as np


"""
将输入的4维张量[batch_size,channel,height,width]中的第一张图像转成numpy矩阵
便于在visdom中进行显示
注意：无论batch_size
"""
def tensor2image(image_tensor):
    image_numpy = image_tensor[0].to("cpu").detach().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    return image_numpy