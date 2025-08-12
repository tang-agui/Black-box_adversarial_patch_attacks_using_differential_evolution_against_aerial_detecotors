
import random
from scipy import rand
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import matplotlib.pyplot as plt


"""
攻击主函数
"""

population_size = 50
generations = 100
F = 0.5
CR = 0.6
xmin = -0.03
xmax = 0.03
OBJ_CONF = 0.4  #   全局变量obj_conf


cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()

# pop size (50, 3 , 608, 608)
# def init_population(dim):

#     population = np.zeros((population_size, dim))
#     for i in range(population_size):
#         for j in range(dim):
#            rand_value = random.random()
#            population[i,j] = xmin + rand_value * (xmax-xmin)
#     return population
#   生成之后的population：50x3x608x608


def init_population(dim):
    # dim_N = dim[0] * dim[1] * dim[2]
    population = []
    # population = np.zeros((population_size, dim[0],dim[1],dim[2]))
    """这里是numpy转tensor再转numpy """
    for i in range(population_size):
        # for j in range(dim):
        rand_value = np.random.rand(dim[0], dim[1], dim[2])
        # rand_value = torch.from_numpy(rand_value).view(dim[0], dim[1], dim[2])
        population_i = xmin + rand_value * (xmax-xmin)
        # population_i = population_i.numpy()
        population.append(population_i)
    # population = torch.tensor(np.array(population))
    return population


def calculate_fitness(taget_image, population):
    """
    计算适应度值
    """
    taget_image = taget_image.cpu().detach().numpy()
    fitness = []
    # function_value = np.zeros(population_size)  # 每个population计算一个值
    model.eval()
    for b in range(population_size):

        attack_image = taget_image + population[b]
        # population_iter_tensor_ = torch.from_numpy(population[b])
        # noise_max, _ = torch.max(population_iter_tensor_.view(1,-1),1)
        # noise_min, _ = torch.min(population_iter_tensor_.view(1,-1),1)
        # print("max : ", noise_max, "min : ", noise_min)
        # attack_image_popu = transforms.ToPILImage('RGB')(population_iter_tensor_.cpu())
        # #   save这里的attack_image观察
        # save_name_wo_reduc = 'fitness_pertur_wo_reduc.png'
        # attack_image_popu.save(save_name_wo_reduc)

        attack_image_ = torch.from_numpy(attack_image)
        # print("max : ", torch.max(attack_image.view(1,-1)))
        attack_image_.clamp_(0, 1)
        attack_image_rgb = transforms.ToPILImage('RGB')(attack_image_.cpu())
        save_name = 'fitness_add_wo_reduc.png'
        attack_image_rgb.save(save_name)
        outputs_boxes = utils.do_detect(
            model, attack_image_rgb, 0.4, 0.4, True)

        f_score = 0.0
        if len(outputs_boxes) == 0:
            fitness.append(-1)
        else:
            outputs_obj_conf = torch.Tensor(outputs_boxes)
            all_obj_conf = outputs_obj_conf[:,5]
            obj_conf_max = max(all_obj_conf)
            f_score = obj_conf_max - OBJ_CONF
            fitness.append(f_score)

    return fitness
    
    '''以下是类别概率输出作为优化loss
    outputs_boxes = utils.do_detect_cls(
            model, attack_image, 0.4, 0.4, True)
        ######################################################################
        #   现在这里返回的是list instances x 15, list，每个值是个tensor
        #   这里有较大的变化，对do_detect()函数进行修改，变成do_detect_cls(),
        #   使得此时得到的是留下检测框的概率部分输出，内部包括get_region_boxes_cls()、
        #   nms_cls().
        #   其中get_region_boxes_cls()此时会将框坐标和概率向量分别输出，用于nms计算，
        #   但是最终仅留下概率向量用于计算
        ######################################################################
        f_score = 0.0
        if len(outputs_boxes) == 0:
            fitness.append(-1)
        else:
            for instance in outputs_boxes:
                # instance = instance.double()
                instance = instance.numpy()
                d = instance[0]
                c = np.min(instance)
                instance.itemset(0, c)
                g = max(instance)
                f_score += (d - g)
            function_value[b] = f_score
            fitness.append(function_value[b])
    '''        
def mutation(population, dim):
    #   population = np.zeros((population_size, dim[0],dim[1],dim[2]))
    Mpopulation = np.zeros((population_size, dim[0], dim[1], dim[2]))
    # dim_N = dim[0] * dim[1] * dim[2]
    for i in range(population_size):

        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        """ 
        以下再对每个元素进行调整
        population[i]: [3,608,608],
        先生成一个和population同维度的随机矩阵, 然后再替换
        元素在xmin和xmax内的保持不变,超出的被替换
        """
        '''
        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = xmin + random.random() * (xmax - xmin)
        '''
        rand_value = np.random.rand(dim[0], dim[1], dim[2])
        population_i = xmin + rand_value * (xmax-xmin)
        # Mpopulation[i] = np.where((Mpopulation[i] >= xmin and Mpopulation[i] <= xmax), Mpopulation[i], population_i)
        Mpopulation[i] = np.where((np.logical_and(
            Mpopulation[i] >= xmin, Mpopulation[i] <= xmax)), Mpopulation[i], population_i)

    return Mpopulation


def crossover(Mpopulation, population, dim):

    Cpopulation = np.zeros((population_size, dim[0], dim[1], dim[2]))
    for i in range(population_size):
        rand_value = np.random.rand(dim[0], dim[1], dim[2])
        Cpopulation[i] = np.where(
            rand_value < CR, Mpopulation[i], population[i])
        
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


def selection(taget_image, Cpopulation, population, pfitness):
    Cfitness = calculate_fitness(taget_image, Cpopulation)  # 更新适应度值
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness


def FDE(clean_image):
    '''
    干净样本和原始标签
    '''
    # dim = clean_image.view(-1,1).size()[0]
    dim = clean_image.size()  # [3,608,608]
    # print("dim : ", dim[0], dim[1], dim[2])
    population = init_population(dim)   # 种群初始化 numpy
    # population_np = torch.Tensor(population)    #tensor
    # print("population size : ", population_np.size())
    # population = population.view(population_size,)
    fitness = calculate_fitness(
        clean_image, population)  # 计算适应度值，适应度值是个和population_size同维的标量
    Best_indi_index = np.argmin(fitness)    # 最小的fitness
    Best_indi = population[Best_indi_index]
    fitness_min = []
    for step in range(generations):
        if min(fitness) < 0:
            break
        #   变异
        fit_min = min(fitness)
        fitness_min.append(fit_min)
        Mpopulation = mutation(population, dim)  # 根据父代生成变异代
        #   交叉
        Cpopulation = crossover(Mpopulation, population, dim)
        #   接下来是选择
        print("step : ", step, "min fitness : ", fit_min)
        population, fitness = selection(
            clean_image, Cpopulation, population, fitness)
        #   求min(fitness)，判断fitness变化趋势
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index]
    np.save("target_set/npy_data_save/" +
        "fitness.npy", fitness_min)
    plt.plot(fitness)
    plt.savefig("fitness curve.png")
    clean_image = clean_image.cpu().detach().numpy()
    final_image = clean_image + Best_indi
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image.clamp_(0, 1)

    return final_image


if __name__ == '__main__':
    images = torch.randn(3, 608, 608)

    images = FDE(images, 1)
