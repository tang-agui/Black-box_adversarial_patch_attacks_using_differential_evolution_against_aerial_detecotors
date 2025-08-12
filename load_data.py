"""
数据处理和补丁添加
Loss计算  # TGJ
"""

import fnmatch
import math
import os
import sys
import time
import gc
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet_v3 import Darknet  # darknet是py文件，定义的是网络架构
import utils
# import edge_extractor_oneStage

ANCHOR_PATH = "data/yolov3_anchors.txt"
DOTA_NAMES = "data/dota.names"


def read_image(path):
    """
    读取一个已经训练好的补丁
    Read an input image to be used as a patch

    :param path: Path to the image to be read.
    :return: Returns the transformed patch as a pytorch Tensor.
    """
    patch_img = Image.open(path).convert('RGB')
    tf = transforms.ToTensor()

    adv_patch_cpu = tf(patch_img)
    return adv_patch_cpu

# def get_boxes_loss(output, anchors, num_anchors):
# torch.nn.MSEloss 有除以N


def bbox_reg(bbox_extractor):
    attack_bbox = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6]).cuda()
    bbox_mse = []
    for box in bbox_extractor:
        # print(len(box))
        box_mse = torch.nn.MSELoss()(attack_bbox, box) * len(box)
        bbox_mse.append(box_mse)
    return bbox_mse


def bbox_decode(output, num_classes, anchors, num_anchors):
    '''   batch x 60 x 19 x 19
              batch x 60 x 38 x 38
              batch x 60 x 76 x 76
    '''

    img_size = (608, 608)
    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)
    stride_h = img_size[1] / h
    stride_w = img_size[0] / w

    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                      for anchor_width, anchor_height in anchors]

    output = output.view(batch*num_anchors, 5+num_classes,
                         h*w)  # batch*3, 20, 19*19
    output = output.transpose(0, 1).contiguous()  # 20, batch*3, 19*19
    output = output.view(5+num_classes, batch *
                         num_anchors*h*w)  # 20, batch*3* 19*19
    grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch *
                                                           num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch *
                                                               num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()

    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    # xs = xs * stride_w
    # ys = ys * stride_h
    # ws = ws * stride_w
    # hs = hs * stride_h

    xs = xs / w
    ys = ys / h
    ws = ws / w
    hs = hs / h  # 这里可进行归一化处理
    output[0] = xs
    output[1] = ys
    output[2] = ws
    output[3] = hs  # 20, batch*3, 19*19

    output = output.view(5+num_classes, batch*num_anchors,
                         h*w)  # 20, batch*3, 19*19
    output = output.transpose(0, 1).contiguous()  # batch*3, 20, 19*19
    output = output.view(batch, num_anchors*(5+num_classes), h, w)

    return output


class MaxProbExtractor(nn.Module):
    """此部分计算YOLO的输出置信度。
    MaxProbExtractor: extracts max class probability for class from YOLO output.
    提取的是输出向量中最大的概率类别，因为最后是要进行优化，使得分类类别不是person
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls):
        """
        cls_id：给定的分类结果id，默认为0，
        num_cls：表示分类数量，在train_patch中调用
        """
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id  # 传入参数
        self.num_cls = num_cls  # 传入参数


    def forward(self, YOLOoutputs):
        #   因为这里不需要筛选，只需要对[x,y,w,h,obj_conf,{classes_conf}]
        #   进行解耦，因此也简单，直接先concatenate
        # attack_bbox = torch.tensor([1e-6,1e-6,1e-6,1e-6])
        #   YOLOoutput：
        '''   batch x 60 x 19 x 19
              batch x 60 x 38 x 38
              batch x 60 x 76 x 76
        '''
        #   '''anchors_step = len(anchors) // num_anchors'''
        anchors = utils.get_anchors(ANCHOR_PATH)
        num_anchors = len(anchors)
        class_names = utils.load_class_names(DOTA_NAMES)
        num_classes = len(class_names)

        output_single_dim = []
        for i, output in enumerate(YOLOoutputs):
            batch = output.size(0)

            h = output.size(2)
            w = output.size(3)  # 32 x 32

            # output = output.view(
            # batch, 5, 5 + self.num_cls, h * w)  #   [batch, num_anchors, x, y, z]
            output = bbox_decode(output, num_classes, anchors[i], num_anchors)
            output = output.view(batch, 3, 5 + self.num_cls, h * w)
            # [batch, 20, 3, 19*19]
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch, 5 + self.num_cls,
                                 3 * h * w)  # [batch, 20, x]
            output_single_dim.append(output)
        #---------------------------------------------------------------------#
        # 以上部分不变
        #---------------------------------------------------------------------#
        # output_cat：[batch, 20, 22743]
        output_cat = torch.cat(output_single_dim, 2)
        
        # 这部分代码实现的是使用sigmoid激活后loss
        
        output_objectness = torch.sigmoid(output_cat[:, 4, :])  # [batch, 3*(19*19+38*38+78*78)]
        max_obj_conf, _ = torch.max(output_objectness, dim=1)  #   这用来确定权重参数
        output_cls_conf = output_cat[:, 5:5 + self.num_cls, :]  # [batch, 15, 22743]
        normal_confs = torch.sigmoid(output_cls_conf)  # 对第二维进行softmax操作, 对类别概率进行softmax激活
        confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 22743]
        max_cls_conf, max_cls_id = torch.max(confs_for_class, dim=1)  #   这用来确定权重参数
      

        return max_cls_conf#, bbox_extract_loss_

        '''
        obj_confs = []
        cls_confs = []
        for i in range(len(max_conf_idx)):

            obj_confs.append(output_objectness[i, max_conf_idx[i]])
            cls_confs.append(confs_for_class[i, max_conf_idx[i]])
        '''

        '''
        bbox_extrac = []
        for i in range(len(max_conf_idx)):
            index = max_conf_idx[i]
            bbox = torch.Tensor([xs[i, index]/w,ys[i, index]/h,ws[i, index]/w,hs[i, index]/h]).cuda()
            bbox_extrac.append(bbox)

        bbox_extract_loss = bbox_reg(bbox_extrac)   #   提取目标检测框的reg_loss
        bbox_extract_loss_ = torch.Tensor(bbox_extract_loss)
        '''

        '''
        obj_confs = torch.Tensor(obj_confs)
        cls_confs = torch.Tensor(cls_confs)  # list都要转tensor
  
        '''


class NPSCalculator(nn.Module):
    """NPSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(
            printability_file, patch_side), requires_grad=False)
        # 获得可打印分数数组，可参考"adv_patch.py"理解
        #  nn.Parameter函数？将一个不可训练的类型转换称可训练的类型

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        # test: change prod for min (find distance to closest color)
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(
            adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(
            adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """用于对补丁进行各种变换
    本类在测试、训练中使用方式不一样
    PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    根据labels中的数据对补丁进行缩放，并将其填充到图像上

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        '''
        self.min_contrast = 1.  # contrast
        self.max_contrast = 1.
        # 光照
        self.min_brightness = 0.
        self.max_brightness = 0.
        # 随机噪声
        self.noise_factor = 0.0
        # 角度

        self.minangle = -0 / 180 * math.pi
        self.maxangle = 0 / 180 * math.pi  # 固定角度
        '''
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False,
            orient=None,
            test_real=False):
        """
        inputs：3_channels adv_patch, 3维标签（已扩维），图片大小
                lab_batch = [batch, max_lab, 5]
        outputs：[1, max_lab, 3, 608, 608]

        # 在训练和测试时传进来的参数是不一样的
        在训练时，lab_batch=(batch, max_lab, 5)，测试时因为是单张图片进行，
        因此在传入时已经进行了扩维，lab_batch=(1, len(labels), 5)，
        即此时传进来的第一维肯定是1（unsqueeze），第二维和labels数据有几行相关
        """

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,300,300], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (608-300) / 2 = 154
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,300,300]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size(
            (lab_batch.size(0), lab_batch.size(1)))  # [batch, max_lab]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [1,2,3,300,300]

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        noise = torch.cuda.FloatTensor(
            adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]
        # 更改代码，添加随机均匀分布噪声
        # Apply contrast/brightness/noise, clamp
        if test_real:  # 测试和训练不同
            adv_batch = adv_batch
        else:
            adv_batch = adv_batch * contrast + brightness + noise  # real_test时固定

        # [1,2,3,300,300]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)  # [1,2,3,1], 最后一个维度扩维
        cls_mask = cls_mask.expand(-1, -1, -1,
                                   adv_batch.size(3))  # [1,2,3,300]
        cls_mask = cls_mask.unsqueeze(-1)

        cls_mask = cls_mask.expand(-1, -1, -1, -1,
                                   adv_batch.size(4))  # [1,2,3,300,300]
        msk_batch_test = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        # print("size of msk_batch_test :", msk_batch_test.size())  # debug
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [1,2,3,608,608]
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 2，旋转角度数量和预测框个数一致
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 300
        lab_batch_scaled = torch.cuda.FloatTensor(
            lab_batch.size()).fill_(0)  # [1,2,5]全0tensor
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size  # [1,2,5]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size  # 第2、3列没用上

        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = 8.0 # 缩放因子
        # pre_scale = 8.0 * 1.414 #   粘贴两个补丁，同时使得面积保持一致大小
        # 要使得面积是1/4，pre_scale是1/2即可
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(1 / pre_scale)) ** 2))  # [1,2]

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # [1, 2]
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))

        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  #   w
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  #   h
        # print("size of target_x :", target_x.size())  # debug
        '''  这部分代码可注释
        '''
        
        if(rand_loc):  # 位置随机
            off_x = targetoff_x * \
                (torch.cuda.FloatTensor(targetoff_x.size()
                                        ).uniform_(-0.2, 0.2))  # 后面的随机太大
            target_x = target_x + off_x  # 加了随机性
            # if target_x < bound_x1:
            #     target_x = bound_x1
            # if target_x > bound_x2:
            #     target_x = bound_x2
            off_y = targetoff_y * \
                (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.2, 0.2))
            # 位置比较关键，想让其偏下，试试参数
            target_y = target_y + off_y
            # if target_y < bound_y1:
            #     target_y = bound_y1
            # if target_y > bound_y2:
            #     target_y = bound_y2

        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [1,2,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        
        if orient == "left":  # 左边
            target_x = target_x - targetoff_x / 6.0  # target_x是中心点坐标
            # 水平4等分或6等分排列
        elif orient == "right":  # 右边
            target_x = target_x + targetoff_x / 6.0

        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        temp_val = adv_batch_t * msk_batch_t

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """向图片上添加补丁
    PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        #  将patch应用到图片上
        # print('img_batch.shape', img_batch.shape)
        # print('adv_batch.shape', adv_batch.shape)
        # 输入[batch, 3, 608, 608], [1, max_lab, 3, 608, 608]
        advs = torch.unbind(adv_batch, 1)
        # 按第二维进行切片，即max_lab，最后advs是一个补丁对那个每个lab的集合
        #  返回指定维度切片后的元组，
        for adv in advs:
            # 对检测框进行遍历
            # adv==0时保留img_batch,否则保留adv（补丁处是补丁，其它地方是原图片）
            # print("adv : ", adv)
            img_batch = torch.where((adv == 0.), img_batch, adv)
            '''
            adv=[1,3,608,608]
            说明：首先这里的img_batch不一定要求维度和adv一致，这应该是torch.where函数的问题
            当img_batch也是[1,3,608,608],返回的也是[1,3,608,608]
            当img_batch是[3,608,608]，返回的也是[1,3,608,608]

            '''
            # 对每一个框进行操作，如果没有框，对应的补丁全是0，所以也不影响
            """
            torch.where(condition,a,b)
            合并a,b两个tensor，满足条件下保留a，否则是b（元素替换）
            """
        return img_batch


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


class InriaDataset(Dataset):
    """读取数据集
    InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        """
        输入参数包括：图片地址、标签地址，最大标签，图片尺寸
        """
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))

        print("n_images = ", n_images, '\n', "n_labels = ", n_labels)  # test

        assert n_images == n_labels, "Number of images and number of labels does't match"
        # 这个地方会做出判断，如果图片数量和标签数不一样则报错
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(
            img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []  # 填入路径
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace(
                '.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        # 分别得到image和lab的路径和名称
        self.max_n_labels = max_lab  # 最大标签数

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace(
            '.jpg', '.txt').replace('.png', '.txt')
        # 分别得到图片和lab的地址
        image = Image.open(img_path).convert('RGB')
        # check to see if label file contains data.
        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        # 若lab为空，直接注释掉行不行？
        else:
            label = np.ones([5])
            # label = np.ones([0])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        # 这个地方对图片进行预处理，所以训练的时候直接使用处理过后的图片就行
        transform = transforms.ToTensor()  # 这个地方转换为tensor，
        image = transform(image)
        # 需要对image和lab进行pad
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img  #
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        # 根据输入图片对原始图片进行resize和padding
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]  # 最终需要填充的labels数据
        if(pad_size > 0):
            # padded_lab = F.pad(lab, (0, 0, 0, pad_size),
            #                    value=1)  # 给labels中的其他数据填充为1
            # 下面行填充为1
            padded_lab = F.pad(lab, (0, 0, 0, pad_size),
                               value=1e-6)  # 给labels中的其他数据填充为1
        else:
            padded_lab = lab
        return padded_lab
        '''
        torch.nn.functional.pad(input, pad, mode, value)
        pad：表示填充方式，分别表示左、右、上、下，此时是下填充
        '''


if __name__ == '__main__':

    # sys.argv=
    '''
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()
    '''
    # img_dir = 'inria/Train/pos'
    # lab_dir = 'inria/Train/pos/yolo-labels'

    img_dir = 'CarData_clean'
    lab_dir = 'CarData_clean/yolo-labels'
    # test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
    #                                           batch_size=3, shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolov2.weights"  # weightfile = "weights/yolov2.weights"
    printfile = "non_printability/30values.txt"

    patch_size = 400

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()

    max_lab_ = 14
    img_size_ = darknet_model.height

    test_loader = torch.utils.data.DataLoader(
        InriaDataset(
            img_dir,
            lab_dir,
            max_lab_,
            img_size_,
            shuffle=True),
        batch_size=8,
        shuffle=True)
    # 数据集测试
    testiter = iter(test_loader)
    images, labels = testiter.next()
    print("images batch size :", images.size(),
          "labels batch size :", labels.size())
    '''返回的label数据是[batch, max_lab, 5]，
    即在数据加载的时候已经对labels数据进行了处理，统一为max_lab'''
    # print('labels data : ', labels)  # 除了正确标签，其它地方都是1填充
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80, cfgfile).cuda()
    nms_calculator = NPSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''以下代码在utils.do_detect()出现，功能一样
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(darknet_model.parameters(), lr=0.0001)

    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ', tl1 - tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open(
            'data/horse.jpg').convert('RGB')  # 将这张图片作为patch进行测试
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)  # 最后一个数，是图片的size
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        # 都是直接对batch进行操作，也就是这些类定义在batch模式上
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(
            img_batch, (darknet_model.height, darknet_model.width))  # 对补丁和图片进行插值过渡
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(
                        obj, 'data') and torch.is_tensor(
                        obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except BaseException:
                        pass
            except BaseException:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1 - t0))
        print('           patch application : %f' % (t2 - t1))
        print('             darknet forward : %f' % (t3 - t2))
        print('      probability extraction : %f' % (t4 - t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4 - t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
