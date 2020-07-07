import cv2
import torch
import numpy as np
import sklearn.metrics as metrics

def calculate_auc(prediction, label):
    result_1D = prediction.flatten()
    label_1D = label.flatten()
    auc = metrics.roc_auc_score(np.asarray(label_1D, dtype=np.int32), result_1D)
    return auc

def calculate_threshold(prediction, label):
    result_1D = prediction.flatten()
    label_1D = label.flatten()
    precision, recall, thresholds = metrics.precision_recall_curve(np.asarray(label_1D, dtype=np.int32),  result_1D, pos_label=1)

    F1_score = 0
    threshold = 0
    for i in range(len(thresholds)):
        temp = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        if temp > F1_score:
            F1_score = temp
            threshold = thresholds[i]
    return F1_score, threshold

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    label = label.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    pre = TP / (TP + FP)
    F1_score = 2 * pre * sen / (pre + sen)
    return acc, sen, F1_score

def test_augment(image, image_size):
    img = np.array(image)
    img = img.squeeze()
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, image_size)

    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]
    img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)

    return torch.Tensor(img5)

def test_augment_pred(pred):
    pred1 = pred[:4] + pred[4:, :, ::-1]
    pred2 = pred1[:2] + pred1[2:, ::-1]
    pred3 = pred2[0] + np.rot90(pred2[1])[::-1, ::-1]
    pred = pred3.copy()
    return pred