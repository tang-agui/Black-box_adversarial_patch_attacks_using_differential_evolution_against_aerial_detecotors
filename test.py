from gettext import find
import numpy
import torch
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import ceil
from torchvision import transforms
from PIL import Image
# a = torch.tensor([3,608,608])
# bb = a.view(-1,1)
# print(a.size(), "bb = ", bb.size())

# random_gene = np.random.random(11111)
# random_view = torch.from_numpy(random_gene).view(1,-1)
# print("random value : ", random_view.size())


# a_list = [0,1,2,3]
# b_min = np.min(a_list)

# print("min in a list : ", b_min)
# nums = np.random.rand(2,3)
# print("nums : ", nums)
# random_nums = np.random.rand(2,3)
# print("random numbers : ", random_nums)
# # nums_copy = copy.deepcopy(nums)
# # nums[nums > 0.5] = random.random()
# # print("after replaced : ", nums)
# # nums_transfer = -0.1+nums *0.2
# nums = np.where(numpy.logical_and(nums>=0.3, nums<=0.8), random_nums,nums)
# # nums = np.where(nums>0.5, random_nums,nums)
# ###################################################################
# # after_replaced = np.where(nums>0.5, random.random(),nums)
# #   此时得到的是同一个随机数
# ###################################################################
# print("after replaced : ", nums)
# print("after transferred : ", nums_transfer)

# a = [1,2,3,54]
# np.save("attacked_img_save/instances_count_save/"+"a.npy",a)

# def return_test():
#     a = 3
#     if a == True:
#         return 1
#     return 0

# value = return_test()
# print("value = ", value)
# population_size = 50
# generations = 100
# F = 0.5
# CR = 0.6
# xmin = -0.15
# xmax = 0.15

# print('hyper-paramenters: {}, generations : {}, xmin : {:.4f}, xmax : {:.4f}'.format(
#     population_size, generations, xmin, xmax))

# print('hyper-paramenters: {}, genetations : {}, xmin : {:.4f}'.format(population_size,generations))

# a = [[1,2,3],[4,5,6]]
# # print(min(a))

# # plt.plot(a)
# # plt.show()
# # plt.savefig('test.png') #   这样可以

# print("a = ", a)
# a_tensor = torch.Tensor(a)
# print("tensor_a : ", a_tensor, a_tensor.size())

# popu_reduc = np.zeros((5, 5))

# print("population reduction : ", popu_reduc[0,:])
# popu_reduc[0:2,:] = np.random.randint(0,609,size=5)
# print("population reduction dim : ", popu_reduc)

# popu_reduc[2:5,:] = np.random.rand(3,5)
# print("population reduction : ", popu_reduc)
# xmin = -0.1
# xmax = 0.1
# RHO = 3
# IMG_dim = 6
# rand_value_origin = np.zeros((3,IMG_dim,IMG_dim))    #   生成Δ

# print("original rand value : ", rand_value_origin)   

# rand_value = xmin + rand_value * (xmax-xmin)

# popu_reduc = np.zeros((RHO, 5))
# popu_reduc[:,0] = np.random.randint(0, IMG_dim, size=RHO)
# popu_reduc[:,1] = np.random.randint(0, IMG_dim, size=RHO)
# print("population location : ", popu_reduc)
# rand_value = np.random.rand(RHO,3)
# # print("rand_value : ", rand_value)
# # print("rand_value plus : ", rand_value * (xmax - xmin))
# population_reduc_i = xmin + rand_value * (xmax - xmin)
# # print("population_reduc_i : ", population_reduc_i)
# popu_reduc[:, 2:5] = population_reduc_i
# # popu_reduc[:, 2:5] = xmin + popu_reduc[:, 2:5] * (xmax-xmin)
# # print("population value = : ", popu_reduc)
# # for j in range(RHO):
# #     print("x : ", int(popu_reduc[j,0]), "y : ", int(popu_reduc[j,1]))
# #     popu_reduc[j, 2:5] = rand_value[:,int(popu_reduc[j,0]), int(popu_reduc[j,1])]
# print("popu_reduc : ",popu_reduc)

# #   还原
# for j in range(RHO):
#     print("jth : ", int(popu_reduc[j,0]),int(popu_reduc[j,1]))
#     rand_value_origin[:,int(popu_reduc[j,0]),int(popu_reduc[j,1])] = popu_reduc[j, 2:5]
#     # print(rand_value_origin)
#     # print("after trans", rand_value_origin[:,int(popu_reduc[j,0]),int(population_reduc_i[j,1])])
# print("after trans : ", rand_value_origin)
# aa = torch.zeros(3,4,4)
# a = torch.rand(2,3)
# print("a = ", a, "\naa : ", aa)
# # a_zeros = torch.zeros_like(a)
# # print("a zeros : ", a_zeros)
# aa[:,0:2,0:3] = a
# print("aa after : ", aa)


#   print(ceil(10.3))
# a = np.ones((2.3,3.3))
# b = np.random.rand(2,2)
# print("a = ", a, "\nb = ", b)
# c = np.kron(a,b)
# print("c = ", c, '\n', "size c : ", c.shape)

# temp_zeros = np.zeros((3, 6, 6),np.uint8)
# temp_rand = np.random.rand(3, 2,2)
# temp_rand_255 = (temp_rand * 255).astype(np.uint8)
# single_patch_ = transforms.ToPILImage(
#             'RGB')(torch.from_numpy(temp_rand).cpu())
# single_patch_.save('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch_.png')
# plt.imshow(temp_rand.transpose(1,2,0))
# plt.show()  
# plt.savefig('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch_plt.png')
# # for k in range(3):
# for i in range(2):
#     for j in range(2):
#         # temp_bbb = temp_rand[k,i,j]
#         # temp_bbb_repeat = temp_bbb.repeat([3],axis=0).repeat([3], axis=1)
#         temp_zeros[0, 3*i:3*(i+1), 3*j:3*(j+1)] = temp_rand_255[0,i,j]
#         temp_zeros[1, 3*i:3*(i+1), 3*j:3*(j+1)] = temp_rand_255[1,i,j]
#         temp_zeros[2, 3*i:3*(i+1), 3*j:3*(j+1)] = temp_rand_255[2,i,j]
# # plt.imshow(temp_zeros.transpose(1,2,0))
# # plt.show()  
# # plt.savefig('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch___.png')
# # print("after replace : ", temp_zeros[1,:,:])
# Image.fromarray(temp_zeros.transpose(1,2,0)).save('test.png')
# a = torch.from_numpy(temp_zeros)
# single_patch = transforms.ToPILImage(
#             'RGB')(torch.from_numpy(temp_zeros).cpu())   #   .transpose(2,1)
# single_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/block/single_patch.png')

# a_list = [[0.1,0.2,0.3], [0.4,0.1],[0.5,0.2,0.1],[0.6,0.1]]
# print(a_list)
# list_len = []
# for items in a_list:
#     # print(len(items))
#     list_len.append(len(items))
# print("len of items : ", list_len)
# a_min = min(list_len)
# # print("a min : ", a_min)
# # print("find len : ", find(list_len == 1))
# # selected = [x for x in list_len if x == 1]
# # print("selected : ", selected)
# selected_index = [i for i,x in enumerate(list_len) if x == a_min]
# print("selected_index : ", selected_index)
# # min_index = np.argmin(list_len)
# # print("argmin value : ", min_index, '\n', "data : ", a_list[min_index])
# select_list = []
# for i in range(len(selected_index)):
#     # print(a_list[selected_index[i]])
#     # print("max value : ", max(a_list[selected_index[i]]))
#     select_list.append(max(a_list[selected_index[i]]))
# print("selected list : ", select_list)
# # print("selected list argmin: ", np.argmin(select_list))

# # print("selected index : ", selected_index[np.argmin(select_list)])
# selected_index_argmin = selected_index[np.argmin(select_list)]
# print("a list index : ", a_list[selected_index_argmin])

# print("min fitness : ", min(a_list))


# a = [-1]
# b = []
# # b.append(-1)

# b.append([-1])
# for item in b:
#     print(len(b))
#     print(len(item))
#     # print("item : ", item[0])

# a = 5.5
# b = 2.5
# c = a if a > b else b
# print("c = ", c)

# PATCH_LIST =[24, 30, 36, 42 ,48]  

# print(PATCH_LIST)
# for i in range(len(PATCH_LIST)):
#     print("number in ", i, ": ", PATCH_LIST[i])
# print(np.exp(1))

#   补丁大小指数分布
# a = np.log(7/4) / 0.23

# b1 = 24 / (np.exp(0.15 * a))
# b2 = 42 / (np.exp(0.38 * a))
# print("coe_1 : ", a, "coe_2 : ", b1, "coe_3 : ", b2)

# def exp_cal(x):
#     b = 2.4331
#     a = 16.6613
#     return a * np.exp(b * x)

# p_3 = exp_cal(0.15)
# p_4 = exp_cal(0.25)
# p_5 = exp_cal(0.38)
# print("p3 : ", p_3,"p4 : ", p_4, "p5 : ", p_5)

# (2023-07-23)如何确定指数函数的参数

# b = np.log(2.5) / 0.3
# print("b = ",b)
# a = 24 / np.exp(0.1*b)
# print("a : ", a)
''''
这里给的逻辑是：
f(0.1) = 24
f(0.4) = 60
然后求解，
但同时还有另外一个问题，指数函数指数部分系数大于0，增加的越快。
'''
#   上下界的确定问题
b = np.log(10) / 0.4
a = 6
print("a : ", a)
print("b = ", b)
''''
这里给的逻辑是：
f(0) = 6
f(0.4) = 60
然后求解，
但同时还有另外一个问题，指数函数指数部分系数大于0，增加的越快。
'''

# b = 
#   现在是想要使得f(0.4)=60左右
x = [0.1,0.2,0.3,0.4]
for i in x:
    y = a * np.exp(i*b)
    print("y = ", y)
