"""
实现逐目标的patch
每个instance正中心放置一个相同的补丁
补丁大小固定
2022-04-26备份，可以完全使用，不需要进行更改

"""

import copy
import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

population_size = 5
ins_dim_sub = 6  # 其中可改变像素数量，小于608
ins_dim_full = 30
generations = 100
IMG_DIM = 608
F = 0.5
CR = 0.6
xmin = 0.
xmax = 1.
OBJ_CONF = 0.4  # 全局变量obj_conf


cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()


def init_population():
    '''
        只初始化个体，粘贴等操作在其它部分实现
    '''
    population = []  # 返回的是[population_size, 3, 608, 608]
    for i in range(population_size):
        # 要是所有扰动相同，则这3行放置在上一个循环即可
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        # rand_value = np.random.rand(ins_dim, ins_dim) #   3通道，或单通道
        population_j = xmin + rand_value * (xmax-xmin)

        population.append(population_j)
    return population


def calculate_fitness(target_image, population, tru_labels):
    """
    """
    fitness = []
    model.eval()
    for b in range(population_size):

        #############################################
        # #   #   采用双线性/最近邻域插值（bilinear/nearest）升维
        population_iter_tensor_ = torch.from_numpy(population[b])
        # single_patch = transforms.ToPILImage(
        #     'RGB')(population_iter_tensor_.cpu())

        # save_name_single = 'testset_attack_100/p_30_bilinear/patches_save/pertur_single.png'
        # single_patch.save(save_name_single)

        # population_deepcopy = copy.deepcopy(population_iter_tensor_)
        # population_iter_tensor = population_deepcopy.unsqueeze(0)
        population_iter_tensor = population_iter_tensor_.unsqueeze(0)
        population_iter = torch.nn.functional.interpolate(
            population_iter_tensor, size=(ins_dim_full, ins_dim_full), mode='nearest')    #   nearest
        # population_iter = torch.nn.functional.interpolate(
        #     population_iter_tensor, size=(ins_dim_full, ins_dim_full), mode='nearest',align_corners=True)  #   bilinear
        population_iter_sque = population_iter.squeeze(0)  # 这里还是tensor[0,1]

        # attack_image_popu = transforms.ToPILImage('RGB')(
        #     population_iter_sque.cpu())  # 这里转为RGB，已经是[0,255]
        population_b = population_iter_sque.numpy()  # 这里将tensor转为numpy()
        #   save这里的attack_image观察
        # save_name = 'testset_attack_100/p_30_bilinear/patches_save/pertur_upsampling.png'
        # attack_image_popu.save(save_name)

        #############################################
        # #   采用kron操作
        # temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
        # # population_b = np.kron(population[b],temp)
        # population_b = np.kron(temp, population[b])   #   上下两行的区别
        # # single_patch = transforms.ToPILImage(
        # #     'RGB')(torch.from_numpy(population[b]).cpu())
        # single_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/patch.png')
        # kron_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(population_b).cpu())
        # kron_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/kron_patch.png')

        #############################################
        #   采用成块补丁，6x6-->30x30
        # temp_zeros = np.zeros((3, ins_dim_full, ins_dim_full))
        # scale_factor = ins_dim_full // ins_dim_sub
        # temp_b = population[b]

        # for i in range(ins_dim_sub):
        #     for j in range(ins_dim_sub):
        #         temp_zeros[:, scale_factor*i:scale_factor *
        #                    (i+1), scale_factor*j:scale_factor*(j+1)] = temp_b[:, i, j].reshape(3, 1, 1)
        #         # temp_zeros[1, scale_factor*i:scale_factor*(i+1), scale_factor*j:scale_factor*(j+1)] = temp_b[1,i,j]
        #         # temp_zeros[2, scale_factor*i:scale_factor*(i+1), scale_factor*j:scale_factor*(j+1)] = temp_b[2,i,j]
        # print("temp zeros : ", temp_zeros)
        # single_patch_255 = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(temp_zeros).cpu())
        # single_patch_255 = Image.fromarray((temp_zeros*255).transpose(1,2,0).astype(np.uint8))
        # single_patch_255.save('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch_255.png')

        # single_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(temp_b).cpu())
        # single_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch.png')

        # population_b = temp_zeros
        #   block
        ##################################

        popu_mask_zeros = np.zeros_like(target_image)

        for j in range(len(tru_labels)):
            w_0 = tru_labels[j][0]  # (x,y)
            h_0 = tru_labels[j][1]
            x_0 = int(w_0 * IMG_DIM)  # (还原到原图片空间)
            y_0 = int(h_0 * IMG_DIM)  # 并取整

            # popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
            #                 x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_b

            x_min = x_0-int(ins_dim_full/2)
            x_max = x_0+int(ins_dim_full/2)
            x_l, x_r = x_min, x_max

            y_min = y_0-int(ins_dim_full/2)
            y_max = y_0+int(ins_dim_full/2)
            y_l, y_r = y_min, y_max

            #   分别对x、y方向进行判断，如果超出图片的大小，则最大值直接设置为图片边界
            if x_max >= IMG_DIM:
                x_max = IMG_DIM
                x_l, x_r = x_max-ins_dim_full, x_max
            if y_max >= IMG_DIM:
                y_max = IMG_DIM
                y_l, y_r = y_max-ins_dim_full, y_max
            if x_min <= 0:
                x_min = 0
                x_l, x_r = x_min, x_min+ins_dim_full
            if y_min <= 0:
                y_min = 0
                y_l, y_r = y_min, y_min+ins_dim_full
            popu_mask_zeros[:, y_l:y_r, x_l:x_r] = population_b

        population_iter_tensor = torch.from_numpy(popu_mask_zeros)
        attack_image = torch.where(
            (population_iter_tensor == 0), target_image, population_iter_tensor)

        # attack_image_instan = transforms.ToPILImage(
        #     'RGB')(population_iter_tensor.cpu())
        # # save这里的attack_image观察
        # save_name_instan = 'testset_attack_100/p_30_bilinear/patches_save/inst_patch.png'
        # attack_image_instan.save(save_name_instan)

        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())  #   这一行必须有
        # #   save这里的attack_image观察
        # save_name_add = 'testset_attack_100/p_30_bilinear/patches_save/ins_patch_add.png'
        # attack_image.save(save_name_add)

        '''
        #   作为分类问题求解
        outputs_boxes = utils.do_detect_cls(
            model, attack_image, 0.4, 0.4, True)
        f_score = 0.0
        if len(outputs_boxes) == 0:
            fitness.append(-1)
        else:
            outputs_cls_conf = torch.Tensor([item.cpu().detach(
            ).numpy() for item in outputs_boxes]).cuda()  # 将list数据转为tensor
            # print("outputs_cls_conf size : ", outputs_cls_conf.size())
            cls_conf_target = outputs_cls_conf[:, 0]  # 得到plane目标的类别概率
            # cls_conf_no_tar = outputs_cls_conf[:, 1:15] #   不指定待攻击类别
            # cls_conf_no_tar_max,_ = torch.max(cls_conf_no_tar, 1) #   剩下的最大值
            # print("cls_conf_no_tar_max : ", cls_conf_no_tar_max)
            # cls_conf_no_tar_max = outputs_cls_conf[:, 1]    #   指定待攻击目标类别为2，
            cls_conf_gap = cls_conf_target - cls_conf_no_tar_max
            max_gap = max(cls_conf_gap)     #  最大的gap都小于0，则问题结束 
            f_score = max_gap
            fitness.append(f_score)

        '''
        #   objectness-confidence作为优化目标
        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)
        
        boxes_attack = []
        for box in outputs_boxes:
            cls_id = box[6]
            if (cls_id == 0):
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes_attack.append(box)
        f_score = 0.0
        if len(boxes_attack) == 0:
            #   2023-08-16
            #   本质上是这里进行了保证，保证了提前终止迭代条件
            #   因为只有这里，fitness的值才会小于0
            fitness.append(-1)
        else:
            outputs_obj_conf = torch.Tensor(boxes_attack)
            all_obj_conf = outputs_obj_conf[:, 4]
            obj_conf_max = max(all_obj_conf)
            #   2023-08-16，以为上面在检测的时候已经使用了0.4阈值
            #   所以这里得到的f_score肯定是大于0的值
            # obj_conf_max = torch.mean(all_obj_conf)
            f_score = obj_conf_max - OBJ_CONF  # 直接使用非sigmoid激活
            #   这里的OBJ_CONF好像可以不要，因为在检测的时候会有阈值
            fitness.append(f_score)
            #   (2023-08-16)——这里计算完后是个标量
            #   而在multi_fit中，计算完后仍是个向量
        
    return fitness


def mutation(population):
    #   population = np.zeros((population_size, dim[0],dim[1],dim[2]))
    Mpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))

    for i in range(population_size):
        r1 = r2 = r3 = 0
        F_temp = random.random()  # 随机均匀的选取F
        # if F_temp > 0.5:
        #     F = 2
        # else:
        #     F = 0.5

        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])
        '''
        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = xmin + random.random() * (xmax - xmin)
        '''
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_i = xmin + rand_value * (xmax-xmin)

        Mpopulation[i] = np.where((np.logical_and(
            Mpopulation[i] >= xmin, Mpopulation[i] <= xmax)), Mpopulation[i], population_i)

    return Mpopulation


def crossover(Mpopulation, population):
    #   dim = population.shape  #
    Cpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))
    for i in range(population_size):
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        Cpopulation[i] = np.where(
            rand_value < CR, Mpopulation[i], population[i])
        # Cpopulation[i] = 0.5 * Mpopulation[i] + 0.5 * population[i] #   PR中的交叉
    '''
    Cpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
            rand_float = random.random()
            if rand_float <= CR:
                Cpopulation[i, j] = Mpopulation[i, j]
            else:
                Cpopulation[i, j] = population[i, j]
    '''
    return Cpopulation


def selection(taget_image, Cpopulation, population, pfitness, tru_label):
    #   这里在计算fitness之前先把Δ'decoder回去
    #   这里的Cpopulation和population都是降维后的空间变量
    # Cpopulation_decoder = decoder(Cpopulation) #  还原回去的变量，还原回去的变量值只用来计算fitness值
    Cfitness = calculate_fitness(taget_image, Cpopulation, tru_label)  # 更新适应度值
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness


def FDE(clean_image, tru_label):
    '''
    干净样本和原始标签
    '''
    #   tru_label转
    dim = clean_image.size()  # [3,608,608]
    population = init_population()   # 种群初始化 numpy

    fitness = calculate_fitness(
        clean_image, population, tru_label)  # 计算适应度值，适应度值是个和population_size同维的标量
    Best_indi_index = np.argmin(fitness)    # 最小的fitness
    Best_indi = population[Best_indi_index]
    fitness_min = []
    for step in range(generations):
        if min(fitness) < 0:
            print("break step : ", step)
            break
        #   变异
        fit_min = min(fitness)
        fit_max = max(fitness)
        fitness_min.append(fit_min)
        Mpopulation = mutation(population)
        #   交叉
        Cpopulation = crossover(Mpopulation, population)
        #   接下来是选择
        print("step : ", step, "min fitness : ", fit_min)
        # print("step : ", step, "max fitness : ", fit_max)
        population, fitness = selection(
            clean_image, Cpopulation, population, fitness, tru_label)
        #   2023-08-16，这里计算完后是个变量，因此可以直接比较fitness的绝对大小
        #   求min(fitness)，判断fitness变化趋势
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index]
        #   这里的Best_indi只有两个维度，下面的decoder需注意
    # np.save("target_set/npy_data_save/" +
    #         "fitness.npy", fitness_min)
    # plt.plot(fitness_min)
    # plt.savefig("fitness curve.png")
    Best_indi_tensor = torch.from_numpy(Best_indi)
    popu_mask_zeros = np.zeros_like(clean_image)
    # clean_image = clean_image.cpu().detach().numpy()
    ######################################################################
    # kron操作
    # temp = np.ones((int(ins_dim_full/ins_dim_sub),
    #                int(ins_dim_full/ins_dim_sub)))
    # population_best = np.kron(temp, Best_indi_tensor)  # 上下两行的区别

    #########################################################################
    #   nearest & bilinear操作
    population_iter_tensor = Best_indi_tensor.unsqueeze(0)
    population_iter = torch.nn.functional.interpolate(
        population_iter_tensor, size=(ins_dim_full, ins_dim_full), mode='nearest')    #   nearest
    # population_iter = torch.nn.functional.interpolate(
    #     population_iter_tensor, size=(ins_dim_full, ins_dim_full), mode='nearest',align_corners=True)  #   bilinear
    population_iter_sque = population_iter.squeeze(0)  # 这里还是tensor[0,1]

    # attack_image_popu = transforms.ToPILImage('RGB')(
    #     population_iter_sque.cpu())  # 这里转为RGB，已经是[0,255]
    population_best = population_iter_sque.numpy()
    
    for j in range(len(tru_label)):
        w_0 = tru_label[j][0]  # (x,y)
        h_0 = tru_label[j][1]
        x_0 = int(w_0 * IMG_DIM)
        y_0 = int(h_0 * IMG_DIM)

        # popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
        #                 x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_best
        
        x_min = x_0-int(ins_dim_full/2)
        x_max = x_0+int(ins_dim_full/2)
        x_l, x_r = x_min, x_max

        y_min = y_0-int(ins_dim_full/2)
        y_max = y_0+int(ins_dim_full/2)
        y_l, y_r = y_min, y_max

        #   分别对x、y方向进行判断，如果超出图片的大小，则最大值直接设置为图片边界
        if x_max >= IMG_DIM:
            x_max = IMG_DIM
            x_l, x_r = x_max-ins_dim_full, x_max
        if y_max >= IMG_DIM:
            y_max = IMG_DIM
            y_l, y_r = y_max-ins_dim_full, y_max
        if x_min <= 0:
            x_min = 0
            x_l, x_r = x_min, x_min+ins_dim_full
        if y_min <= 0:
            y_min = 0
            y_l, y_r = y_min, y_min+ins_dim_full
        popu_mask_zeros[:, y_l:y_r, x_l:x_r] = population_best

    final_pertur = torch.from_numpy(popu_mask_zeros)
    final_image = torch.where(
        (final_pertur == 0.), clean_image, final_pertur)
    # final_image = clean_image + popu_mask_zeros   #
    # final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image.clamp_(0, 1)

    return final_image


if __name__ == '__main__':
    images = torch.randn(3, 608, 608)

    images = FDE(images, 1)
