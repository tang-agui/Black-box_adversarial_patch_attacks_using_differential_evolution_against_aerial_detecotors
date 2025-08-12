import utils_self
import fnmatch
import os
import torch
import numpy as np


# imgdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_trainset/clean'  
# labdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_trainset/clean/yolo-labels'  # 到labels层

# imgdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_testset/clean'  
# labdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_testset/clean/yolo-labels'  # 到labels层

imgdir = 'transfer_attack_YOLOv4/plane/patched_pre'  
labdir = 'transfer_attack_YOLOv4/plane/yolo-labels'  # 到labels层

# imgdir = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/images'
# labdir = "/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels"

n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images : ", n_images)

n_lab = len(fnmatch.filter(os.listdir(labdir), '*.txt'))

length_txt_boxes = utils_self.txt_len_read(labdir)
#   分别返回总instance和每个文件的instance
# index = (len_ins_account[:] > 100)
# num_instances = np.array(len_ins_account)
# np.save('instances_account/num_instances.npy', num_instances)
# utils_self.hist_draw(len_ins_account,"instances_account/instances_account_01.png")
# filename = open('instances_account/num_instances.txt', 'w')
# a = sum(i <= 6 for i in len_ins_account)
# print("a = ", a)
# for value in len_ins_account:
#     filename.write(str(value))
#     filename.write('\n')
# filename.close()
#------------------------------------------------------------#
#   还需要得到最大的labels
#------------------------------------------------------------#
print("Total instances : ", length_txt_boxes)
# print("max length of labels : ", max(len_ins_account))
# print("num of instances per image : ", len_ins_account)
print("Total txt file : ", n_lab)

n_device = torch.cuda.device_count()
# print("num of cudas : ", n_device)
