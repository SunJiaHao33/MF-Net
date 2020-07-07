import os
import numpy as np
import sklearn.metrics as metrics

from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
AUC = AverageMeter()
AUC2 = AverageMeter()
label_dir = 'E:/project/pycharm/MF-net/dataset/DRIVE/test/1st_manual/'
pred_dir = 'E:/project/pycharm/MF-net/outputs/results/DRIVE/Net_50/07-06 15-52-16'
mask_dir = 'E:/project/pycharm/MF-net/dataset/DRIVE/test/mask/'

label_files = os.listdir(label_dir)

for num in range(len(label_files)):
    label = Image.open(os.path.join(label_dir, '{:02d}_manual1.gif'.format(num + 1)))
    pred = Image.open(os.path.join(pred_dir, '{:02d}_test-mask.png'.format(num + 1)))
    mask = Image.open(os.path.join(mask_dir, '{:02d}_test_mask.gif'.format(num + 1)))

    label = label.resize((576, 576))
    pred = pred.resize((576, 576))
    mask = mask.resize((576, 576))

    label = np.asarray(label)
    pred = np.asarray(pred)
    mask = np.asarray(mask)

    label_list = []
    pred_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 255:
                label_list.append(label[i][j])
                pred_list.append(pred[i][j])
    auc2 = metrics.roc_auc_score(label.flatten(), pred.flatten())
    auc = metrics.roc_auc_score(np.asarray(label_list), np.asarray(pred_list))
    AUC.update(auc)
    AUC2.update(auc2)
    print(auc, auc2)


print('final', AUC.avg, AUC2.avg)