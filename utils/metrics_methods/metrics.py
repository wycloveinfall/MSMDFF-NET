import numpy as np
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Precision(self):
        self.precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return self.precision

    def Pixel_Recall(self):
        self.recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return self.recall

    def Pixel_F1(self):
        f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) #
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _generate_matrix_one(self, gt_image, pre_image):
        '''
        date:2022.08.30
        fct:返回单个样本的混淆矩阵
        author:wyc
        '''
        confusion_matrix_ = np.zeros((gt_image.shape[0],self.num_class,self.num_class)).astype('int')
        for index,(_gt_image,_pre_image) in enumerate(zip(gt_image, pre_image)):
            mask = (_gt_image >= 0) & (_gt_image < self.num_class) #
            label = self.num_class * _gt_image[mask].astype('int') + _pre_image[mask].astype('int')
            count = np.bincount(label, minlength=self.num_class**2)
            # print('len(count):',len(count))
            confusion_matrix_[index] = count.reshape(self.num_class, self.num_class)
            # confusion_matrix_1 = confusion_matrix_.sum((0))
        return confusion_matrix_

    def add_batch_and_return_iou(self, gt_image, pre_image):
        '''
        date:2022.08.30
        fct:在原有的add batch基础上，返回单个样本的Iou，
        author:wyc
        '''
        assert gt_image.shape == pre_image.shape
        confusion_matrix_ = self._generate_matrix_one(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix_.sum((0))
        IoU_batch = np.zeros((gt_image.shape[0]))
        for index,c_m in enumerate(confusion_matrix_):
            IoU_batch[index] = c_m[1, 1] / (c_m[1, 1] + c_m[1, 0] + c_m[0, 1] + 1e-10)
        return IoU_batch

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_one_batch_iou(self,gt_image, pre_image):
        pre_image = pre_image.cpu().data.numpy().squeeze()
        gt_image = gt_image.cpu().data.numpy().squeeze().astype(np.uint8)
        pre_image[pre_image < 0.5] = 0
        pre_image[pre_image >= 0.5] = 1
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        IoU = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1] + 1e-10)
        return IoU





