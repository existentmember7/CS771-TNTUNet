import cv2
import glob
import os.path as osp
import numpy as np
import random

def demo_video():
    colors_path = "F:\\KITTI_new\\dataset\\val\\color"
    colors_paths = glob.glob(osp.join(colors_path, "*.png"))
    pred_path = "..\\results\\this"
    pred_paths = glob.glob(osp.join(pred_path, "*.png"))

    img_array = []
    for i in range(len(colors_paths)):
        color = cv2.imread(colors_paths[i])
        pred = cv2.imread(pred_paths[i])
        pred = cv2.resize(pred, (color.shape[1],color.shape[0]), interpolation = cv2.INTER_AREA)
        final = np.append(color, pred, axis=0)
        size = (final.shape[1],final.shape[0])
        img_array.append(final)

    out = cv2.VideoWriter('CS766_project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def generate_colors(class_num):
    random.seed(42)
    np.random.seed(42)
    RGB_list = []

    for i in range(class_num):
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)
        RGB_list.append([R,G,B])

    RGB_list = np.array(RGB_list)

    return RGB_list

def align_cls_colors():
    target_dir = "F:\\KITTI_new\\dataset\\val\\label\\000102.png"
    RGB_list = generate_colors(12)

    target = cv2.imread(target_dir)
    img = np.zeros_like(target)
    for i in range(len(RGB_list)):
        img[target[:,:,0] == i, :] = RGB_list[i,:]
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("label.png", img)

align_cls_colors()