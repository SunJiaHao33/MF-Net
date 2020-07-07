import os
import torch
import argparse
import torch.optim as optim
import torchvision.transforms as transforms

from time import time
from torch.autograd import Variable
from loss import dice_bce_loss

from data import ImageFolder
from network.ResidualUnet import DeepResUNet
from network.model import Net_34, Net_50, Net_101, Net_50_FCK, Net_50_MFP, Net_50_baseline, Net_50_add
from utils.tools import second2time, make_save_file
from utils.logger import inial_logger

def train(args, model, train_loader, test_loader, criterion, optimizer):
    lr = args.lr
    best_loss = 100

    # 保存权重以及logger的位置
    epoch_save_dir = make_save_file(os.path.join('./outputs/weights', args.dataset, model._get_name()))  # 这个函数按时间创建文件夹
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)

    # 初始化logger以保存日志
    logger = inial_logger(os.path.join(epoch_save_dir, 'logger.log'))
    logger.info('[{}] epoch:{} lr:{} batch_size:{} image_size:{} save_interval:{}'.format(
        args.dataset, args.total_epoch, args.lr, args.batch_size, args.image_size, args.save_interval))


    for epoch in range(args.total_epoch):
        start_time = time()
        epoch_loss = 0

        # 训练
        for iter, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = Variable(images.cuda())
            labels = Variable(labels.cuda())

            outputs = model(inputs)
            loss = criterion(labels, outputs)
            loss.backward()
            epoch_loss += float(loss.item())
            optimizer.step()

        epoch_loss = epoch_loss/len(train_loader)
        logger.info('[train] epoch:{} train_loss:{:.6f} lr:{} time:{}'.format(epoch, epoch_loss, lr, second2time(time() - start_time)))

        # 测试
        if args.test:
            epoch_loss = 0
            model.eval()
            start_time = time()
            for iter, (images, labels) in enumerate(test_loader):
                inputs = Variable(images.cuda())
                labels = Variable(labels.cuda())

                outputs = model(inputs)
                loss = criterion(labels, outputs)
                epoch_loss += float(loss.item())

            epoch_loss = epoch_loss/len(test_loader)
            logger.info('[test] epoch:{} test_loss:{:.6f} lr:{} time:{}'.format(epoch, epoch_loss, lr, second2time(time() - start_time)))


        # 保存权重
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(epoch_save_dir, 'epoch-{}.pth'.format(epoch)))

        # 每次迭代loss没有下降
        if epoch_loss >= best_loss:
            no_optim += 1
        else:
            no_optim = 0
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(epoch_save_dir, 'best_result.pth'))

        if no_optim > args.num_early_stop:
            logger.info('[stop] early stop at %d epoch =================' % epoch)
            break

        # 跳出局部最优
        if no_optim > args.num_update_lr:
            no_optim = 0
            if lr < 5e-7:
                logger.info('[break] learning rate: {} at epoch {} ============='.format(lr, epoch))
                break
            model.load_state_dict(torch.load(os.path.join(epoch_save_dir, 'best_result.pth')))
            
            lr = lr/2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info('[lr change] learning rate: {} --> {} at epoch {} ========='.format(lr*2, lr, epoch))


def main(args):
    x_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])

    # 数据集
    train_dataset = ImageFolder(datasets=args.dataset, image_size=args.image_size, transform=x_transforms, target_transform=y_transforms)
    test_dataset = ImageFolder(datasets=args.dataset, image_size=args.image_size, mode='test', transform=x_transforms, target_transform=y_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 模型及参数
    model = Net_50_baseline()
    model.cuda()
    criterion = dice_bce_loss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # 训练
    train(args, model=model,  train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    dataset = 'DRIVE'  # selected in [DRIVE, CHASEDB1, STARE]

    if dataset == 'DRIVE':
        parse.add_argument('--dataset', default='DRIVE', type=str)
        parse.add_argument('--batch_size', default=2, type=int)
        parse.add_argument('--image_size', default=(576, 576), type=int)

    elif dataset == 'CHASEDB1':
        parse.add_argument('--dataset', default='CHASEDB1', type=str)
        parse.add_argument('--batch_size', default=1, type=int)
        parse.add_argument('--image_size', default=(960, 960), type=int)

    elif dataset == 'STARE':
        parse.add_argument('--dataset', default='STARE', type=str)
        parse.add_argument('--batch_size', default=2, type=int)
        parse.add_argument('--image_size', default=(640, 640), type=int)

    # 全局参数
    parse.add_argument('--test', default=False, type=bool)
    parse.add_argument('--total_epoch', default=100, type=int)
    parse.add_argument('--lr', default=0.0002, type=float)

    parse.add_argument('--num_early_stop', default=20, type=int)
    parse.add_argument('--num_update_lr', default=10, type=int)
    parse.add_argument('--save_interval', default=50, type=int)
    args = parse.parse_args()

    main(args)
