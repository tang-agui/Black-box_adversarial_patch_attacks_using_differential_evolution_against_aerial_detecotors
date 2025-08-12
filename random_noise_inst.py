import time
import numpy as np
import torch
import torchvision.transforms as transforms
from darknet_v3 import Darknet
import os
import utils_self
import utils
from PIL import Image


# xmin = -0.3
# xmax = 0.3
ins_dim = 30 

imgdir = '/mnt/share1/tangguijian/Data_storage/large_vehicle/wo_L_vehicle_0.1filter/testset_100/img'
clean_labdir = "/mnt/share1/tangguijian/Data_storage/large_vehicle/wo_L_vehicle_0.1filter/testset_100/yolo-labels"
savedir = "larger_vehicle_attack_100/random_noise"

cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()
img_size = model.height

print('per ins dim : {:.4f}'.format(ins_dim))

print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))

class_names = utils.load_class_names('data/dota.names')
total_count = 0.
total_count_attacked = 0.
for imgfile in os.listdir(imgdir):
    print("new image")  #
    rand_noise = np.random.rand(3, ins_dim, ins_dim)
    # rand_noise = xmin + rand_value * (xmax-xmin)    #   每张图片生成一个扰动
    
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
        w, h = img.size
        print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new(
                    'RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new(
                    'RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize((img_size, img_size))
        padded_img = resize(padded_img)
        
        tru_lab = utils.read_truths_pre_7(txtpath)
        images_to_tensor = utils_self.img_transfer(padded_img)
        
        popu_mask_zeros = np.zeros_like(images_to_tensor)
        
        for j in range(len(tru_lab)):
            w_0 = tru_lab[j][0]  # (x,y)
            h_0 = tru_lab[j][1]
            x_0 = int(w_0 * img_size)  # (还原到原图片空间)
            y_0 = int(h_0 * img_size)    #   并取整

            popu_mask_zeros[:, y_0-int(ins_dim/2):y_0+int(ins_dim/2),
                            x_0-int(ins_dim/2):x_0+int(ins_dim/2)] = rand_noise
            
        #   生成和img同维度的噪声
        rand_noise_tensor = torch.from_numpy(popu_mask_zeros)
        img_noised = torch.where((rand_noise_tensor == 0.), images_to_tensor, rand_noise_tensor)
        # img_noised = images_to_tensor + rand_noise_tensor
        img_noised.clamp_(0, 1) #   clamp到[0,1]间
        img_noised_pre = transforms.ToPILImage(
                'RGB')(img_noised.cpu())  # 转RGB图片
        
        save_name_add = name + '.png'
        # img_noised_pre.save(save_name_add)
        noised_dir = os.path.join(
                savedir, 'img_patched/', save_name_add)
        img_noised_pre.save(noised_dir)
        
        boxes_cls = utils.do_detect(
            model, img_noised_pre, 0.4, 0.4, True)
            # boxes_cls_attack = utils.nms(boxes_cls_attack, 0.4)
        boxes = []  # 定义对特定类别的boxes
        for box in boxes_cls:
            cls_id = box[6]
            if (cls_id == 5):
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes.append(box)
                    
        tru_lab = utils.read_truths_pre_7(txtpath)
        total_count += len(tru_lab)
        total_count_attacked += len(boxes)
        
        pre_name = name + ".png"  # 不添加后缀
        pre_dir = os.path.join(
            savedir, 'patched_pre/', pre_name)

        utils.plot_boxes(img_noised_pre, boxes, pre_dir,
                         class_names=class_names)  # 这里画出来的预测框也是经过筛选后

        txtpath_write = os.path.abspath(os.path.join(
            savedir, 'yolo-labels/', txtname))
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