import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.ATC import ATC
from utils.data import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import os
import cv2


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, best_loss):
    model.train()
    loss_record1,loss_record2, loss_record3, loss_record4,loss_record5 = AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter() ,AvgMeter()
    accum = 0
    for i, (images, gts) in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        lateral_map_5,lateral_map_4, lateral_map_3, lateral_map_2,lateral_map_1 = model(images)

        # ---- loss function ----
        
        loss5 = structure_loss(lateral_map_5, gts)
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record1.update(loss1.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)
        loss_record5.update(loss5.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f},[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}],[lateral-5: {:.4f},'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'ATC-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'ATC-%d.pth'% epoch)
    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./data', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='ACTNet')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()

    # ---- build models ----
    model = ATC(pretrained=True).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = './data/EORSSD-dataset-master/EORSSD/train-images/'
    gt_root = './data/EORSSD-dataset-master/EORSSD/train-labels/'


    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,trainsize_H=384,trainsize_W=384)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss)
