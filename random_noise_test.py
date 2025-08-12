import time
import numpy as np
import torch
import torchvision.transforms as transforms
from darknet_v3 import Darknet
import os
import utils_self
import utils


noise_factor = 0.5
imgdir='random_noise/imgs'
clean_labdir = "random_noise/imgs/yolo-labels"
savedir = "random_noise/inst_patch_test"

cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()
class_names = utils.load_class_names('data/dota.names')
total_count = 0.
total_count_attacked = 0.
for imgfile in os.listdir(imgdir):
    print("new image")  #
    t_single_begin = time.time()
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        print("image file path is ", imgfile)
        img = utils_self.load_image_file(imgfile)  # 注意这里的不同
        #   单张图片预测格式
        img = utils_self.img_transfer(img).cuda()
        
        noise = torch.cuda.FloatTensor(
                    img.size()).uniform_(-1, 1) * noise_factor
        # print("L2 : ", torch.norm(noise), "L-inf : ", torch.norm(noise, p='inf'))
        # noise_max, _ = torch.max(noise.view(1,-1),1)
        # noise_min, _ = torch.min(noise.view(1,-1),1)
        # print("max : ", noise_max, "min : ", noise_min)
        # attack_image_popu = transforms.ToPILImage('RGB')(noise.cpu())
        #   save这里的attack_image观察
        # save_name = 'noise_save.png'
        # attack_image_popu.save(save_name)
        #   生成和img同维度的噪声
        img_noised = img + noise
        img_noised.clamp_(0, 1) #   clamp到[0,1]间
        img_noised_pre = transforms.ToPILImage(
                'RGB')(img_noised.cpu())  # 转RGB图片
        
        # save_name_add = 'noise_save_add.png'
        # img_noised_pre.save(save_name_add)
        boxes_cls = utils.do_detect(
            model, img_noised_pre, 0.4, 0.4, True)
            # boxes_cls_attack = utils.nms(boxes_cls_attack, 0.4)
        boxes = []  # 定义对特定类别的boxes
        for box in boxes_cls:
            cls_id = box[6]
            if (cls_id == 0):
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes.append(box)
                    
        tru_lab = utils.read_truths_pre_7(txtpath)
        total_count += len(tru_lab)
        total_count_attacked += len(boxes)
        
        pre_name = name + ".png"  # 不添加后缀
        pre_dir = os.path.join(
            savedir, 'pre_results/', pre_name)

        utils.plot_boxes(img_noised_pre, boxes, pre_dir,
                         class_names=class_names)  # 这里画出来的预测框也是经过筛选后

        txtpath_write = os.path.abspath(os.path.join(
            savedir, 'pre_results/', 'yolo-labels/', txtname))
        textfile = open(txtpath_write, 'w+')
        #   需要注意，此时写入的是攻击后图片的预测结果，不再只写入预测为飞机的instances
        for box in boxes:
            textfile.write(
                f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        textfile.close()
attacked_count = total_count - total_count_attacked
print("total instances in GT : ", total_count, "instances after attacks : ", total_count_attacked)
print('Accuracy of attack: %f %%' %
          (100 * float(attacked_count) / total_count))