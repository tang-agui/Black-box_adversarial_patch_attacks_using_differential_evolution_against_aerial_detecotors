
"""
(2022-08-15)
发展自适应算法
"""

"""
实现逐目标的patch
每个instance正中心放置一个相同的补丁
补丁大小固定

测试不同的fitness函数，在计算fitness值时不是取max，而是取所有instance输出
然后再通过两步，比较长度和最大值最小，对population进行筛选

"""

# import matplotlib.pyplot as plt
# from PIL import Image

import random
from cv2 import PARAM_ALGORITHM
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
population_size = 5
ins_dim_sub = 6  # 其中可改变像素数量，小于608
#   ins_dim_full = 30    #   ins_dim_full = 6，此时分析复制补丁块的作用。
generations = 100
IMG_DIM = 608
F = 0.5
CR = 0.6
xmin = 0.
xmax = 1.
OBJ_CONF = 0.4  # 全局变量obj_conf
PATCH_LIST = [30, 30, 42, 42, 60]  # 设置不同的补丁大小，分别对应不同的检测框
#   这里是经验性设置的参数

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
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_j = xmin + rand_value * (xmax-xmin)

        population.append(population_j)
    return population


def patch_size_def(box_size_larger):
    """
    input: box_sie_larger，经过筛选后的检测框的较大边
    根据检测框的大小确定缩放完后补丁的完整大小——ins_dim_full 
    """

    if box_size_larger <= 0.15:  # 或者这里的0.2设置为0.15？？？
        ins_dim_full = PATCH_LIST[0]  # patch_size = 24

    elif box_size_larger <= 0.25:
        ins_dim_full = PATCH_LIST[1]  # patch_size = 30

    elif box_size_larger <= 0.38:
        ins_dim_full = PATCH_LIST[2]  # patch_size = 36 / 42

    # elif box_size_larger <= 0.45:
    #     ins_dim_full = PATCH_LIST[3]

    else:
        ins_dim_full = PATCH_LIST[4]  # patch_size = 60
    return ins_dim_full


def calculate_fitness(target_image, population, tru_labels):
    """
    在根据目标大小进行adaptive调整时，需要根据tru_labels进行设置缩放因子
    tru_labels = [x,y,w,h,obj_conf,cls_conf, id]
    """
    n_query_iteration = 0
    fitness = []
    model.eval()
    for b in range(population_size):
        #   （1）定义（归一化）缩放因子：f(x)=10x+3
        #   （2）定义分段函数——因为阵列复制的时候要求阵列复制完后的补丁大小为6的整数倍，因此
        #   采用分段函数似乎更合理
        #   采用kron操作
        """
        #   统一大小补丁
        temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
        population_b = np.kron(temp, population[b])   #   上下两行的区别
        """
        # single_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(population[b]).cpu())
        # single_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/patch.png')
        # kron_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(population_b).cpu())
        # kron_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/kron_patch.png')

        #   根据每个目标的大小设置不同大小的补丁

        popu_mask_zeros = np.zeros_like(target_image)

        for j in range(len(tru_labels)):

            x_00 = tru_labels[j][0]  # (x,y)——得到目标的中心位置
            y_00 = tru_labels[j][1]
            #   取的是每个标签的[x,y]坐标
            x_0 = int(x_00 * IMG_DIM)  # (还原到原图片空间)
            y_0 = int(y_00 * IMG_DIM)  # 并取整

            w_0 = tru_labels[j][2]
            h_0 = tru_labels[j][3]  # 得到目标的宽、高
            #   adaptive scale
            larger_edge = w_0 if w_0 > h_0 else h_0  # 取较大边作为缩放因子

            ins_dim_full = patch_size_def(larger_edge)  # 得到此时完整的补丁大小

            patch_scale = int(ins_dim_full / ins_dim_sub)
            temp = np.ones((patch_scale, patch_scale))
            population_b = np.kron(temp, population[b])  # 实现阵列复制

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
            # PIL读取的图片为[C,H,W]格式
            popu_mask_zeros[:, y_l:y_r, x_l:x_r] = population_b

        population_iter_tensor = torch.from_numpy(popu_mask_zeros)
        attack_image = torch.where(
            (population_iter_tensor == 0), target_image, population_iter_tensor)
        # attack_image = target_image + population_iter_tensor  # 注意加法前后的数据类型一致

        # attack_image_instan = transforms.ToPILImage(
        #     'RGB')(population_iter_tensor.cpu())
        #  # save这里的attack_image观察
        # save_name_instan = 'patch_adap_scale_test/patch_saves/inst_patch.png'
        # attack_image_instan.save(save_name_instan)
        # # attack_image = torch.from_numpy(attack_image)
        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())
        #   save这里的attack_image观察
        # save_name_add = 'patch_adap_scale_test/patch_saves/ins_patch_add.png'
        # attack_image.save(save_name_add)

        #   objectness-confidence作为优化目标
        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)
        
        n_query_iteration += 1
        
        boxes_attack = []
        for box in outputs_boxes:
            cls_id = box[6]
            if (cls_id == 5):   #   id === 0 --> plane, id == 5 --> large vehicle, id == 6 --> ship
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes_attack.append(box)
        f_score = 0.0

        if len(boxes_attack) == 0:
            fitness.append([-1])
        else:
            outputs_obj_conf = torch.Tensor(boxes_attack)
            all_obj_conf = outputs_obj_conf[:, 4]
            # obj_conf_max = max(all_obj_conf)
            # f_score = obj_conf_max - OBJ_CONF  # 直接使用非sigmoid激活
            #   这里的OBJ_CONF好像可以不要，因为在检测的时候会有阈值
            f_score = all_obj_conf - OBJ_CONF
            fitness.append(f_score)

    return fitness, n_query_iteration


def mutation(population):
    #   population = np.zeros((population_size, dim[0],dim[1],dim[2]))
    Mpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))

    for i in range(population_size):
        r1 = r2 = r3 = 0
        F_temp = random.random()  # 随机均匀的选取F
        if F_temp > 0.5:
            F = 2
        else:
            F = 0.5

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
    Cfitness, n_queries = calculate_fitness(taget_image, Cpopulation, tru_label)  # 更新适应度值
    for i in range(population_size):
        if len(Cfitness[i]) <= len(pfitness[i]):  # Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness, n_queries


def fitness_selection(fitness):

    fitness_len = []
    for items in fitness:
        if (len(items) == 1 and items[0] == -1):
            fitness_min_value = -1
            fitness_len.append(len(items))
        else:
            fitness_len.append(len(items))
    fitness_min_len = min(fitness_len)  # 找到最小的长度值

    selected_index = [i for i, x in enumerate(
        fitness_len) if x == fitness_min_len]  # 对于fitness_min_len不等于1的情况需要重新设计
    #   找到长度为最小值的所有个体
    select_list = []  # 里面的元素是标量
    for i in range(len(selected_index)):
        select_list.append(max(fitness[selected_index[i]]))
    fitness_index = selected_index[np.argmin(select_list)]  # 返回index就可以
    #   找到其中值最小的index，记为此时的个体
    fitness_index_value = fitness[fitness_index]  # 找到此时个体对应的fitness值
    fitness_min_value = max(fitness_index_value)  # 并找到这个个体对应的instance的最大值
    return fitness_index, fitness_min_value


def FDE(clean_image, tru_label):
    '''
    干净样本和原始标签
    '''
    Total_single_image_queries = 0   #   定义攻击单张图片所需要的所有queries
    #   tru_label转
    population = init_population()   # 种群初始化 numpy

    fitness, n_queries_init = calculate_fitness(
        clean_image, population, tru_label)
    Total_single_image_queries += n_queries_init
    #   计算适应度值，适应度值是个和population_size同维的向量
    '''   这里的fitness是向量还是标量就和fitness function的设计相关
    1) 如果为max操作，则得到的为与population size同维度的标量
    2) 如果为multi_fitness操作，则得到与population size x tru_label同维的向量
    '''
    fitness_index, fitness_min_value = fitness_selection(fitness)
    #   fitness_index用于确定索引，fitness_min_value用于条件判断
    Best_indi = population[fitness_index]

    for step in range(generations):
        if fitness_min_value < 0:
            print("break step : ", step)
            break
        #   变异
        Mpopulation = mutation(population)
        #   交叉
        Cpopulation = crossover(Mpopulation, population)
        #   接下来是选择
        print("step : ", step, "min fitness : ", fitness_min_value)

        population, fitness, n_queries = selection(
            clean_image, Cpopulation, population, fitness, tru_label)
        #   这里根据父代和offspring代进行筛选，得到子代，
        fitness_index, fitness_min_value = fitness_selection(fitness)
        #   然后对子代中的个体进行筛选
        Best_indi = population[fitness_index]
        Total_single_image_queries += n_queries
        #   这里的Best_indi只有两个维度，下面的decoder需注意

    Best_indi_tensor = torch.from_numpy(Best_indi)
    popu_mask_zeros = np.zeros_like(clean_image)
    # clean_image = clean_image.cpu().detach().numpy()
    ######################################################################
    # kron操作
    # temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
    # population_best = np.kron(temp, Best_indi_tensor)   #   上下两行的区别

    #########################################################################
    #   block操作
    # population_best = np.zeros((3, ins_dim_full, ins_dim_full))
    # scale_factor = ins_dim_full // ins_dim_sub
    # # temp_b = population[b]

    # for i in range(ins_dim_sub):
    #     for j in range(ins_dim_sub):
    #         population_best[:, scale_factor*i:scale_factor*(
    #             i+1), scale_factor*j:scale_factor*(j+1)] = Best_indi[:, i, j].reshape(3, 1, 1)

    for j in range(len(tru_label)):
        w_0 = tru_label[j][0]  # (x,y)
        h_0 = tru_label[j][1]
        x_0 = int(w_0 * IMG_DIM)
        y_0 = int(h_0 * IMG_DIM)

        # popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
        #                 x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_best
        w_0 = tru_label[j][2]
        h_0 = tru_label[j][3]  # 得到目标的宽、高

        larger_edge = w_0 if w_0 > h_0 else h_0  # 取较大边作为缩放因子

        ins_dim_full = patch_size_def(larger_edge)  # 得到此时完整的补丁大小

        patch_scale = int(ins_dim_full / ins_dim_sub)
        temp = np.ones((patch_scale, patch_scale))
        population_best = np.kron(temp, Best_indi_tensor)  # 实现阵列复制

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

    return final_image, Total_single_image_queries


if __name__ == '__main__':
    images = torch.randn(3, 608, 608)

    images = FDE(images, 1)
