import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.ATC import ATC
from utils.data import test_dataset
import imageio
import cv2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./snapshots/ACTNet/ACT-XXX.pth')
    parser.add_argument('--test_path', type=str,
                        default='./testmaps', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='./testmaps', help='path to save inference segmentation')
    parser.add_argument('--save_path1', type=str, default='./testmaps1', help='path to save inference segmentation')
    parser.add_argument('--save_path2', type=str, default='./testmaps2', help='path to save inference segmentation')

    opt = parser.parse_args()

    model = ATC().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)


    image_root = '/home/baoliuxin/TransFuse-main/data/EORSSD-dataset-master/EORSSD/test-images/'
    gt_root = '/home/baoliuxin/TransFuse-main/data/EORSSD-dataset-master/EORSSD/test-labels/'
    test_loader = test_dataset(image_root, gt_root, testsize_H=384, testsize_W=384)



    for i in range(test_loader.size):
        image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)


        image = image.cuda()

        with torch.no_grad():
            #_,_,res2, res1, res = model(image)
            _,_,_,_, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()


        if opt.save_path is not None:
            cv2.imwrite(opt.save_path+'/' + name, res*255)



