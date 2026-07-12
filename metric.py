

import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        # Eq. (24): Precision = TP / (TP + FP)
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + self.eps)
        return precision

    def Recall(self):
        # Eq. (25): Recall = TP / (TP + FN)
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)
        return recall

    def F1(self):
        # Eq. (23): F1 = 2 * Precision * Recall / (Precision + Recall)
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp + self.eps)
        Recall = tp / (tp + fn + self.eps)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall + self.eps)
        return F1

    def OA(self):
        # Eq. (26): OA = (TP + TN) / (TP + TN + FN + FP)
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        # Eq. (27): IoU = TP / (TP + FN + FP)
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU

    def Dice(self):
       
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        return Dice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
