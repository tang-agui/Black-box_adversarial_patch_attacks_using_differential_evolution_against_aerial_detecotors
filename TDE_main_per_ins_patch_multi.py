"""
实现逐目标的patch
每个instance正中心放置一个补丁
补丁大小固定
补丁各异，独立进化
patch的数量和原始图片中的instances数量相关（每个instance对应一个补丁）
"""

import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

population_size_base = 5
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


def init_population(tru_labels):
    '''
        只初始化个体，粘贴等操作在其它部分实现
    '''
    population = []
    population_size = population_size_base * len(tru_labels)
    for i in range(population_size):

        rand_value = np.float32(np.random.rand(
            3, ins_dim_sub, ins_dim_sub))  # 要是所有扰动相同，则这3行放置在上一个循环即可
        population_j = xmin + rand_value * (xmax-xmin)
        population.append(population_j)

    return population  # (len(tru_label)*5, 3, 6, 6)


def calculate_fitness(target_image, population, tru_labels):
    """
    """
    fitness = []
    model.eval()
    # population_size = 5 * len(tru_labels)
    for i in range(population_size_base):
        popu_mask_zeros = np.zeros_like(target_image)
        #   会根据基础的population size计算相应数量的fitness
        #   第1个instance粘贴前五个，第2个instance粘贴第6-10个
        for j in range(len(tru_labels)):
            b = population_size_base * j + i
            w_0 = tru_labels[j][0]  # (x,y)
            h_0 = tru_labels[j][1]
            x_0 = int(w_0 * IMG_DIM)  # (还原到原图片空间)
            y_0 = int(h_0 * IMG_DIM)  # 并取整

            temp = np.ones((int(ins_dim_full/ins_dim_sub),
                           int(ins_dim_full/ins_dim_sub)))
            # population_b = np.kron(population[b],temp)
            population_b = np.kron(temp, population[b])

            popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
                            x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_b

        population_iter_tensor = torch.from_numpy(popu_mask_zeros)
        # attack_image_instan = transforms.ToPILImage(
        #     'RGB')(population_iter_tensor.cpu())
        # save这里的attack_image观察
        # save_name_instan = 'target_set/instan_patch/patch_test_save/patch_per_inst/inst_patch.png'
        # attack_image_instan.save(save_name_instan)

        attack_image = torch.where(
            (population_iter_tensor == 0.), target_image, population_iter_tensor)

        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())
        #   save这里的attack_image观察
        # save_name_add = 'target_set/instan_patch/patch_test_save/patch_per_inst/ins_patch_add.png'
        # attack_image.save(save_name_add)

        #   objectness-confidence作为优化目标
        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)

        #   假设此时不进行筛选，则攻击的是所有的目标
        boxes_attack = []
        for box in outputs_boxes:
            cls_id = box[6]
            if (cls_id == 0):
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes_attack.append(box)
        f_score = 0.0
        if len(boxes_attack) == 0:
            fitness.append([-1])
            
        else:
            outputs_obj_conf = torch.Tensor(boxes_attack)
            # [0.589,0.5939,0.5958,0.5941,0.5961]
            all_obj_conf = outputs_obj_conf[:, 4]
            # obj_conf_max = max(all_obj_conf)
            # f_score_max = obj_conf_max - OBJ_CONF  #    此时的f_score是动态的
            f_score = all_obj_conf - OBJ_CONF
            #   这里的OBJ_CONF好像可以不要，因为在检测的时候会有阈值
            fitness.append(f_score)
            # fitness_max.append(f_score_max)

    return fitness  


def mutation(population, tru_label):

    population_size = population_size_base * len(tru_label)
    Mpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))

    for i in range(population_size_base):
        #   也是对每一个instance对应的（5个）population进行交叉
        r1 = r2 = r3 = 0
        F_temp = random.random()  # 随机均匀的选取F
        if F_temp > 0.5:
            F = 2
        else:
            F = 0.5
        for j in range(len(tru_label)):
            b = population_size_base * j + i

            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(population_size_base * j,
                                    population_size_base * (j+1) - 1)
                r2 = random.randint(population_size_base * j,
                                    population_size_base * (j+1) - 1)
                r3 = random.randint(population_size_base * j,
                                    population_size_base * (j+1) - 1)
            Mpopulation[b] = population[r1] + F * \
                (population[r2] - population[r3])

    for j in range(population_size):

        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_i = xmin + rand_value * (xmax-xmin)

        Mpopulation[j] = np.where((np.logical_and(
            Mpopulation[j] >= xmin, Mpopulation[j] <= xmax)), Mpopulation[j], population_i)
        #   这里对population进行范围再限制，得到的结果应该是所有的Mpopulation [0，1]
    return Mpopulation


def crossover(Mpopulation, population, tru_label):
    #   dim = population.shape  #
    population_size = population_size_base * len(tru_label)
    Cpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))

    for i in range(population_size_base):

        for j in range(len(tru_label)):
            b = population_size_base * j + i

            rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
            population_j = xmin + rand_value * (xmax-xmin)

            Cpopulation[b] = np.where(
                population_j < CR, Mpopulation[b], population[b])

    return Cpopulation


def selection(taget_image, Cpopulation, population, pfitness, tru_label):

    Cfitness = calculate_fitness(taget_image, Cpopulation, tru_label)  # 更新适应度值
    for i in range(population_size_base):
        for j in range(len(tru_label)):
            b = population_size_base * j + i
            if len(Cfitness[i]) <= len(pfitness[i]):
                population[b] = Cpopulation[b]
                pfitness[i] = Cfitness[i]
            else:
                population[b] = population[b]
                pfitness[i] = pfitness[i]
    return population, pfitness


def fitness_selection(fitness, tru_labels):
    # if len(tru_labels) == 1:
    #     #   如果只有一个instance，则直接找到最小值
    #     fitness_index = np.argmin(fitness)
    #     fitness_min_value = fitness[fitness_index]    #   得到此时索引的值
    #     #   对于单instance情况，此时fitness_min_len < 0 即 break
    # else:
    fitness_len = []
    for items in fitness:
        if (len(items) == 1 and items[0] == -1):
            fitness_min_value = -1
            fitness_len.append(len(items))
        else:
            fitness_len.append(len(items))
    fitness_min_len = min(fitness_len)
    
    selected_index = [i for i, x in enumerate(
        fitness_len) if x == fitness_min_len]   #   对于fitness_min_len不等于1的情况需要重新设计
    
    select_list = []
    for i in range(len(selected_index)):
        select_list.append(max(fitness[selected_index[i]]))
    fitness_index = selected_index[np.argmin(select_list)]  #   返回index就可以
    fitness_index_value = fitness[fitness_index]
    fitness_min_value = max(fitness_index_value)
    return fitness_index, fitness_min_value


def FDE(clean_image, tru_label):
    '''
    干净样本和原始标签
    '''
    #   tru_label转
    #   dim = clean_image.size()  # [3,608,608]
    population = init_population(tru_label)   # 种群初始化 numpy

    fitness = calculate_fitness(
        clean_image, population, tru_label)  #
    # print("fitness : ", fitness, '\n', "fitness max : ", fitness_max)
    fitness_best = []
    
    fitness_index, fitness_min_value = fitness_selection(fitness, tru_label)
    #   fitness_index用于确定索引，fitness_min_value用于条件判断
    # print("index : ", fitness_index)
    for i in range(len(tru_label)):
        #   需要根据每个instance返回最佳的个体
        best_index = population_size_base*i + fitness_index
        # print("best index : ", population_size_base*i + fitness_index)
        fitness_best_indi = population[best_index]
        fitness_best.append(fitness_best_indi)

    # Best_indi_index = np.argmin(fitness)    # 最小的fitness
    # Best_indi = population[Best_indi_index]
    for step in range(generations):
        if fitness_min_value < 0:
            #   此时判断条件也需要修改，修改为当长度小于0时停止
            print("break step : ", step)
            break
        #   变异

        Mpopulation = mutation(population, tru_label)
        #   交叉
        Cpopulation = crossover(Mpopulation, population, tru_label)
        #   接下来是选择
        print("step : ", step, "min fitness : ", fitness_min_value)
        population, fitness = selection(
            clean_image, Cpopulation, population, fitness, tru_label)

        fitness_best = []  # 再初始化
        fitness_index, fitness_min_value = fitness_selection(fitness, tru_label)
        for i in range(len(tru_label)):

            best_index = population_size_base*i + fitness_index
            fitness_best_indi = population[best_index]
            fitness_best.append(fitness_best_indi)

    popu_mask_zeros = np.zeros_like(clean_image)
    #   会根据基础的population size计算相应数量的fitness
    #   第1个instance粘贴前五个，第2个instance粘贴第6-10个
    for j in range(len(tru_label)):
        w_0 = tru_label[j][0]  # (x,y)
        h_0 = tru_label[j][1]
        x_0 = int(w_0 * IMG_DIM)  # (还原到原图片空间)
        y_0 = int(h_0 * IMG_DIM)  # 并取整

        temp = np.ones((int(ins_dim_full/ins_dim_sub),
                       int(ins_dim_full/ins_dim_sub)))
        # population_b = np.kron(population[b],temp)
        population_b = np.kron(temp, fitness_best[j])

        popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
                        x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_b

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
