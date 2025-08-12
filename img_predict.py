import time
import numpy as np
import torchvision.transforms as transforms
from darknet_v3 import Darknet
import os
import utils_self
import utils
from PIL import Image
import fnmatch


# imgdir = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/images'
imgdir = 'test_img_save/imgs'
# clean_labdir = "target_set/yolo-labels"

savedir = "test_img_save"
labels_GT = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels'    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfgfile = "/mnt/share1/tangguijian/Black_AE_Evo/cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()

count = 0
total_count = 0
net_correct = 0

img_size = model.height
img_width = model.width
class_names = utils.load_class_names('/mnt/share1/tangguijian/Black_AE_Evo/data/dota.names')

t0 = time.time()
for imgfile in os.listdir(imgdir):
    print("new image")  #
    t_single_begin = time.time()
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        # txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

        # open beeld en pas aan naar yolo input size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        print("image file path is ", imgfile)
        img = utils_self.load_image_file(imgfile)  # 注意这里的不同

        w, h = img.size
        # print("original w = ", w, ", h = ", h)
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
        #   单张图片预测格式

        boxes_cls = utils.do_detect(model, padded_img, 0.4, 0.4, True)
        # boxes_cls = utils.nms(boxes_cls, 0.4)

        boxes_attack = []  # 定义对特定类别的boxes
        for box in boxes_cls:
            cls_id = box[6]
            if (cls_id == 0):  # # id=0-->plane，id=5-->large_vehilce，id=6-->ship
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes_attack.append(box)

        pre_name = name + ".png"  # 不添加后缀
        pre_dir = os.path.join(savedir, 'patched_pre/', pre_name)

        utils.plot_boxes(padded_img, boxes_attack, pre_dir,
                            class_names=class_names)  # 这里画出来的预测框也是经过筛选后
        txtpath_pre_clean = os.path.abspath(
            os.path.join(savedir, 'yolo-labels', txtname))
        textfile = open(txtpath_pre_clean, 'w+')  #

        
        for box in boxes_attack:
            textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        textfile.close()
        
labdir_pre = os.path.abspath(os.path.join(savedir, 'yolo-labels/'))
n_lab_pre = len(fnmatch.filter(os.listdir(labdir_pre), '*.txt'))
length_txt_boxes_pre = utils_self.txt_len_read(labdir_pre)
print("Total instances pre: ", length_txt_boxes_pre)
print("Total txt file pre: ", n_lab_pre)

n_lab_GT = len(fnmatch.filter(os.listdir(labels_GT), '*.txt'))
length_txt_boxes_GT = utils_self.txt_len_read(labels_GT)
print("Total instances GT : ", length_txt_boxes_GT)
print("Total txt file GT : ", n_lab_GT)

attacked_count = length_txt_boxes_GT - \
    length_txt_boxes_pre  # ground truth和检测到gap
print("total instances ground truth: ", length_txt_boxes_GT, "instances after attack : ",
      length_txt_boxes_pre, "total instances gap : ", attacked_count)
print('Accuracy of attack: %f %%' %
      (100 * float(attacked_count) / length_txt_boxes_GT))

t1 = time.time()
t_cost = (t1-t0)/60
print("Processing Done!")
print("Total Running Time : ", t_cost, "minutes !")
