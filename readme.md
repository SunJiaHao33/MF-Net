# MF-net

## train:
训练的参数已经封装到每个parse了，修改dataset可以直接run

模型在main函数中 修改model=Net_50()即可修改模型

（CHASE-DB1、STARE 数据集是我自行分的训练和测试集，之后会再完善随机抽取的代码）

## test:
修改dataset，选择模型model，更改load_weight_path模型权重文件地址

run生成预测图片，并计算相关指标


## DRIVE 
![avatar](figure/DRIVE.png)

F1_Score:0.808921 <big>**Sensitivity:0.808921 Accuracy:0.966477 AUC:0.982722**</big>

## CHASE-DB1
FOV means field of view masks

CHASE-DB1 共28张标注样本，随机选取20张用于训练，剩下8张用于测试
![avatar](figure/CHASE-DB1.png)

<big>**F1_Score:0.811520  Sensitivity:0.829131 Accuracy:0.975698 AUC:0.988860**</big>

## STARE 
STARE 共20张标注样本，随机选取16张用于训练，剩下4张用于测试
![avatar](figure/STARE.png)
F1_Score:0.791935 <big>**Sensitivity:0.809211  Accuracy:0.974139**</big> AUC:0.986207 


