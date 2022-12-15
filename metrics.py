import os, sys
import argparse
import glob

from sklearn.metrics import f1_score, confusion_matrix
import cv2
import numpy as np
from statistics import mean


parser = argparse.ArgumentParser(description="Low-light Image Enhancement")
parser.add_argument(
    "--gt",
    default="",
    type=str,
    help="path to ground truth (default: none)",
)
parser.add_argument(
    "--predict",
    default="",
    type=str,
    help="path to predictions (default: none)",
)
parser.add_argument(
    "--nclass",
    default=11,
    type=int,
    help="number of class",
)
parser.add_argument(
    "--average",
    default="macro",
    type=str,
    help="average typr for f1",
)


class Metrics:
    def __init__(self, gt_path, pred_path, num_class, average_type="macro"):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.num_class = num_class
        self.average_type = average_type
        self.f1_avg = 0
        self.f1_min = 0
        self.f1_max = 0
        self.f1_all = []
        self.miou_avg = 0
        self.miou_min = 0
        self.miou_max = 0
        self.miou_all = []
        self.weighted_mIoU = 0

        self.compute()

    def compute(self):
        gt_imgs = [os.path.basename(x) for x in glob.glob(os.path.join(self.gt_path, '*.png'))]
        pred_imgs = [os.path.basename(x) for x in glob.glob(os.path.join(self.pred_path, '*.png'))]
        gt_set, pred_set = set(gt_imgs), set(pred_imgs)
        diff_set = gt_set - pred_set
        if len(diff_set) != 0:
            print(f"ground truth and prediction do not match. Exit!")
            sys.exit(-1)

        labels = [x for x in range(self.num_class + 1)]
        for img_name in gt_imgs:
            gt = cv2.imread(os.path.join(self.gt_path, img_name))[:,:,0]
            gt = gt.flatten()
            pred = cv2.imread(os.path.join(self.pred_path, img_name))[:,:,0]
            pred = pred.flatten()

            # f1 score
            self.f1_all.append((f1_score(gt, pred, average=self.average_type, zero_division=0), img_name))

            # mIoU
            gt_count = np.bincount(gt)
            pred_count = np.bincount(pred)
            conf_matrix = confusion_matrix(gt, pred, labels=labels)

            I = np.zeros(self.num_class + 1)
            U = np.zeros(self.num_class + 1)
            for i in range(self.num_class + 1):
                I[i] = conf_matrix[i][i]
            U = gt_count + pred_count - I
            IoU = I / U
            mIoU = np.nanmean(IoU)
            weighted_mIoU = self.weighted_sum(gt_count, IoU)
            self.miou_all.append((mIoU, img_name))

            self.weighted_mIoU += weighted_mIoU

        self.weighted_mIoU/=len(gt_imgs)

        self.f1_all.sort(key=lambda x: x[0])
        self.f1_avg = mean([x[0] for x in self.f1_all])
        self.f1_max = self.f1_all[-1]
        self.f1_min = self.f1_all[0]

        self.miou_all.sort(key=lambda x: x[0])
        self.miou_avg = mean([x[0] for x in self.miou_all])
        self.miou_max = self.miou_all[-1]
        self.miou_min = self.miou_all[0]
    
    def weighted_sum(self, gt_count, IoU):
        total = np.sum(gt_count)
        ans = 0
        for i in range(len(gt_count)):
            if ~np.isnan(IoU[i]):
                ans += IoU[i] * gt_count[i]

        return ans/total


    def print(self):
        print(f"weighted_mIoU = {self.weighted_mIoU}")
        print(f"f1_max = {self.f1_max[0]} ({self.f1_max[1]}), f1_min = {self.f1_min[0]} ({self.f1_min[1]}), f1_avg = {self.f1_avg}")
        print(f"miou_max = {self.miou_max[0]} ({self.miou_max[1]}), miou_min = {self.miou_min[0]} ({self.miou_min[1]}), miou_avg = {self.miou_avg}")

    def printAll(self):
        print(f"all f1: {self.f1_all}")
        print(f"all miou: {self.miou_all}")


if __name__ == "__main__":
    args = parser.parse_args()
    metrics = Metrics(args.gt, args.predict, args.nclass, args.average)
    metrics.print()
    # metrics.printAll()
