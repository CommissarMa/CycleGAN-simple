from torchvision import transforms
from PIL import Image

"""
在加载数据集的过程中会使用到，进行各种变换
"""
def data_transform(opt, grayscale=False, convert_to_tensor=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if opt.resize_or_crop == 'resize_and_crop':#缩放+裁剪
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':#裁剪
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if convert_to_tensor:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]#因为这里做了normalize处理，值域为[-1,1]。所以不能图片不能直接展示了，待会要做处理转回来。
    return transforms.Compose(transform_list)