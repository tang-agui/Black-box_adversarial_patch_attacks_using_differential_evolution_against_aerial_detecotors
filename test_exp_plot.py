'''
分别得到label数据中的宽、高数据
目的是要给出分布图
'''
import os
import numpy as np
import matplotlib.pyplot as plt 
#   plane
imgdir = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/images'
clean_labdir = "/mnt/jfs/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels"
#   format: [x,y,w,h,obj_conf, cls_prob, id]
# savedir = "plane_attack_100/random_noise_adap_scale"


    # for instances calculate
len_txt = 0
# len_ins_account = []
# labels_w = []
# labels_h = []
# max_len = []
# for txtfile_label in os.listdir(clean_labdir):  # 得到所有文件名
#     if txtfile_label.endswith('.txt'):
        
#         txtfile = os.path.abspath(os.path.join(clean_labdir, txtfile_label)) 
        
#         # if os.path.getsize(txtfile):
#         #     myfile = np.loadtxt(txtfile)
#         #     myfile = open(txtfile)
#         #     file_data = myfile.readlines()
#         #     if len(myfile) == 1:
#         #         labels_w.append(myfile[2])
#         #         print("data[2]", myfile[2])
#         #         labels_h.append(myfile[3])
            
#         #     else:
#         #         for data in myfile:
#         #             print("data in data : ", data)
#         #             labels_w.append(data[2])
#         #             print("data[2]", data[2])
#         #             labels_h.append(data[3])

#         if os.path.getsize(txtfile):
#                 myfile = open(txtfile)
#                 file_items = myfile.readlines()     #   这样读进来的格式类似list
#                 if len(file_items):
#                     for item in file_items:
#                         w = float(item.rsplit()[2])
#                         h = float(item.rsplit()[3])
#                         max_length = max(w,h)
#                         labels_w.append(float(item.rsplit()[2]))
#                         # print("data[2]", float(item.rsplit()[2]))
#                         labels_h.append(float(item.rsplit()[3]))
#                         max_len.append(max_length)

# np.savetxt("adap_scale/label_w.txt",labels_w)
# np.savetxt("adap_scale/label_h.txt",labels_h)
# np.savetxt("adap_scale/max_length.txt",max_len)


labels_w = np.loadtxt('adap_scale/label_w.txt')
labels_h = np.loadtxt('adap_scale/label_h.txt')

x_length = len(labels_w)
print(" length of x : ", x_length)


plot_hist = plt.hist(labels_w, bins=30, density=True,label=r'The distribution of $x$')

# plt.plot(plot_hist[1][0:30],plot_hist[0],"r",linewidth=2) 

plt.xlim([0.05, 0.95])
plt.ylim([0,5])
plt.xlabel(r'x')
plt.ylabel('Number of instances')
plt.legend()
plt.savefig('adap_scale/label_w.png',bbox_inches='tight',pad_inches=0.1)

plt.figure()
plt.hist(labels_h, bins=50, density=True)
plt.savefig('adap_scale/label_h.png')

# plt.figure()
# plt.hist(max_len, bins=30, density=True)
# plt.savefig('adap_scale/max_len.png')

