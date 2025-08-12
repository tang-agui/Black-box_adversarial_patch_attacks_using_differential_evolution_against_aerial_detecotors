'''
使用预测得到的标签分别计算recall
'''


import os
import fnmatch
import utils_self
import time


t0 = time.time()

# pre_label_dir = 'testset_attack_100/p_30_kron_scalable_F/yolo-labels'  # 粘贴补丁后的预测结果
# labdir_gro_tru = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels'

# pre_label_dir = 'testset_attack_100/p_30_direct/yolo-labels'  # 粘贴补丁后的预测结果
# labdir_gro_tru = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels'

# pre_label_dir = 'ship_attack_100/p_30_kron_multi_fitness/yolo-labels'  # 粘贴补丁后的预测结果
pre_label_dir = 'transfer_attack_Faster_rCNN/ship/yolo-labels'  # 粘贴补丁后的预测结果
labdir_gro_tru = '/mnt/share1/tangguijian/Data_storage/ship_608/wo_ship_0.1filter/testset_100/yolo-labels'

# pre_label_dir = 'transfer_attack_Faster_rCNN/plane/yolo-labels'  # 粘贴补丁后的预测结果
# labdir_gro_tru = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels'

n_gro_tru_labels = len(fnmatch.filter(os.listdir(labdir_gro_tru), '*.txt'))
n_pre_labels = len(fnmatch.filter(os.listdir(pre_label_dir), '*.txt'))

print("total ground truth labels : ", n_gro_tru_labels)
print("total predicted labels : ", n_pre_labels)
length_gro_tru = utils_self.txt_len_read(labdir_gro_tru)
print("length of ground truth labels : ", length_gro_tru)
length_pre_label = utils_self.txt_len_read(pre_label_dir)
print("length of predicted labels : ", length_pre_label)

conf_thresh = 0.4  # threshold = 0.4  # 0.4和0.6差别较大
iou_thresh = 0.1    #   default 0.5
precision, recall, ASR = utils_self.eval_list( 
    pre_label_dir, labdir_gro_tru, conf_thresh, iou_thresh)
ASR_1 = (length_gro_tru - length_pre_label) / length_gro_tru
# print("final precision : ", precision)
# print("final recall : ", recall)
print("final ASR_1 : ", ASR_1)
print("ASR calculated from recall : ", ASR)
t1 = time.time()
t11 = (t1-t0)/60
print('recall & precision Total running time: {:.4f} minutes'.format(t11))
