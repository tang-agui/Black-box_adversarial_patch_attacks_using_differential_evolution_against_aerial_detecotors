This paper propose a novel framework for black-box adversarial patch attacks against aerial imagery object detectors using 
differential evolution (DE). Specifically, we first propose a dimensionality reduction strategy to address the dimensionality 
curse in high-dimensional optimization problems and improve optimization efficiency. Then, we design three universal fitness
 functions to help DE find promising solutions within a limited computational budget according to the diverse outputs of the 
 detectors. Finally, we conduct extensive experiments on the DOTA dataset against state-of-the-art object detectors such as YOLOv3,
  YOLOv4, and Faster R-CNN. Results show that our method exhibits superior performance in addressing black-box attacks on 
  aerial imagery object detection. To the best of our knowledge, this is the first work to explore the use of DE in black-box 
  adversarial patch attacks against aerial imagery object detectors.

This project is an official implementation of the paper "Black-box adversarial patch attacks using differential evolution against aerial
imagery object detectors", which has been published in Engineering Applications of Artifical Intelligence.
对应本文博士大论文第三章。


- 差分进化算法（Differntial Evolution）的各种参数，例如种群大小，迭代次数等都在TDE_main_*.py（以TDE_main_ins_patch_multi_fit.py为例）函数中定义，具体包括：

population_size = 5
ins_dim_sub = 6  # 其中可改变像素数量，小于608
ins_dim_full = 30    #   ins_dim_full = 6，此时分析复制补丁块的作用。
generations = 100
IMG_DIM = 608
F = 0.5
CR = 0.6
xmin = 0.
xmax = 1.
OBJ_CONF = 0.4  # 全局变量obj_conf
可根据需要进行修。

- 模型权重文件和配置文件：（以TDE_main_ins_patch_multi_fit.py为例）
    - cfgfile = "cfg/yolov3-dota.cfg"
    - weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"
与之前项目保持一致。需要使用绝对路径。

- 训练数据等参数：
在TFNS_*.py（以TFNS.py为例）函数中定义，具体包括：
    - imgdir = '/mnt/jfs/tangguijian/Data_storage/Black_AE_Evo_testset/images'
    - clean_labdir = "/mnt/jfs/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels"
    为待攻击的数据集和标签，仍然在服务器中以绝对路径定义。
    需要注意的是，本项目攻击与具体的目标类别有关，在论文中攻击了不同的类别，包括plane，ship（/mnt/jfs/tangguijian/Data_storage/ship_608）和large-vehicle（/mnt/jfs/tangguijian/Data_storage/large_vehicle）
    - savedir = "training_patches_test/patched_imgs_testing"
    需要注意的是，因为要保存攻击完后的图片，以及相应的标签、检测结果，因此还需要手动在savedir下新建子目标，具体包括：
        - img_patched:
        - patched_pre:
        - yolo-labels: 
        自定义。

- 补丁攻击（main）
运行命令：
nohup python -u TFNS.py > training_patches_test/training_logs/Evo_patch_attack_testing.log 2>&1 &
运行该命令，在相应log文件中出现：
image file path is  /mnt/jfs/tangguijian/Data_storage/Black_AE_Evo_testset/images/P1397__1__4064___1016.png
step :  0 min fitness :  tensor(0.5994)
step :  1 min fitness :  tensor(0.5993)
step :  2 min fitness :  tensor(0.5994)
step :  3 min fitness :  tensor(0.5994)
step :  4 min fitness :  tensor(0.5995)
step :  5 min fitness :  tensor(0.5995)
即表示程序运行正常，等候程序运行结束即可。


本project为黑盒攻击场景，本文件夹针对的是YOLOv3的黑盒攻击，因此使用的仍然是
本project已经在服务器上完成验证，具体使用的是刘旭博士账号下的/home/ubuntu/anaconda3/envs/patchAT虚拟环境。
- YOLOv3
对应的具体服务器路径为：/mnt/jfs/tangguijian/Black_AE_Evo_DOTA_yolov3。
- Faster R-CNN 
本方法可扩展到其它检测器中，例如Faster R-CNN，对应的服务器路径为：/mnt/jfs/tangguijian/AerialDetection_black
但是攻击Faster R-CNN时，需要不一样的虚拟环境运行Faster R-CNN，具体可进入该项目了解。
- RetinaNet
还可攻击RetinaNet，但是仍然依赖攻击RetinaNet的检测器环境，具体可进入了解：/mnt/jfs/tangguijian/s2anet_Evo_BB


本工程默认使用差分进化算法（Differntial Evolution），但是同样可使用其它进化算法，例如GA：TFNS-GA-attack.py，在对应paper中的ablation_study章节有。

