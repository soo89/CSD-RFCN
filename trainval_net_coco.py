# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pdb
import pprint
import time
import math
import _init_paths
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from skimage.transform import rescale, resize, downscale_local_mean

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb, rank_roidb_ratio

import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc_0712_semi', type=str)
    parser.add_argument('--arch', dest='arch', default='rfcn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=21, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="save",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda', default = True,
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs', default = True,
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--ohem', dest='ohem',
                        help='Use online hard example mining for training',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=15, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=None, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',          # faster_rcnn_1_20_10021
                        help='checkepoch to load model',
                        default=12, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',  #faster_rcnn_1_28_2504
                        help='checkpoint to load model',
                        default=10021, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def rampweight(iteration, total_epoch = 21, data_len = 10022):
    ramp_up_end = data_len * 2 * 3
    ramp_down_start = data_len * (6 * 3 + 2) #+  2 * data_len
    iter_max = total_epoch * data_len

    beta = 1

    # if (iteration < 100):
    #     ramp_weight = 0
    if (iteration < ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2)) * beta
    elif (iteration > ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (iter_max - iteration) / (iter_max - ramp_down_start)), 2)) * beta
    else:
        ramp_weight = 1 * beta

    if (iteration == 0):
        ramp_weight = 0

    return ramp_weight

if __name__ == '__main__':

    args = parse_args()

    if args.arch == 'rcnn':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet
    elif args.arch == 'rfcn':
        ########## consistency loss !!!!!!!!!!!!!!!!!!  #########
        from model.rfcn.resnet_atrous_consistency import resnet
        # from model.rfcn.resnet_atrous import resnet
    elif args.arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712_semi":
        args.imdb_name = "voc_2007_trainval"
        args.imdb_name_unlabel = "voc_2012_trainval" #+coco_2014_train+coco_2014_val"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    unlabel_imdb, unlabel_roidb, unlabel_ratio_list, unlabel_ratio_index = combined_roidb(args.imdb_name_unlabel)
    unlabel_train_size = len(unlabel_roidb)

    save_coco_unflip = unlabel_roidb[0]
    save_coco_flip = unlabel_roidb[23079]

    with open("/home/user/JISOO/R-FCN.pytorch-master/data/coco/voc_included.txt") as f:
        lines = f.readlines()

    coco_roidb = []
    coco_flip_roidb = []


    for a in range(len(lines)):
        save_coco_unflip['image'] ="/home/user/JISOO/R-FCN.pytorch-master/data/coco/images/" + lines[a][:-1]
        save_coco_flip['image'] ="/home/user/JISOO/R-FCN.pytorch-master/data/coco/images/" + lines[a][:-1]
        img = cv2.imread(save_coco_unflip['image'])
        height, width, channels = img.shape
        save_coco_unflip['width'] = width
        save_coco_unflip['height'] = height
        coco_roidb.append(save_coco_unflip)
        coco_flip_roidb.append(save_coco_flip)

    coco_roidb = coco_roidb + coco_flip_roidb

    unlabel_roidb = unlabel_roidb + coco_roidb

    unlabel_ratio_list, unlabel_ratio_index = rank_roidb_ratio(unlabel_roidb)




    print('{:d} roidb entries'.format(len(roidb)))
    print('{:d} roidb entries'.format(len(unlabel_roidb)))
    # print('{:d} roidb entries'.format(len(coco_unlabel_roidb)))

    output_dir = os.path.join(args.save_dir, args.arch, args.net, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supervised_batch_size = 1
    unsupervised_batch_size = args.batch_size - supervised_batch_size

    sampler_batch = sampler(train_size, supervised_batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, supervised_batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=supervised_batch_size, sampler=sampler_batch, num_workers=args.num_workers, drop_last=True)

    unlabel_sampler_batch = sampler(unlabel_train_size, unsupervised_batch_size)
    unlabel_dataset = roibatchLoader(unlabel_roidb, unlabel_ratio_list, unlabel_ratio_index, unsupervised_batch_size, imdb.num_classes, training=True)
    unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=unsupervised_batch_size, sampler=unlabel_sampler_batch, num_workers=args.num_workers, drop_last=True)



    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    semi_check = torch.ByteTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        semi_check = semi_check.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    semi_check = Variable(semi_check)

    # initilize the tensor holder here.
    unlabel_im_data = torch.FloatTensor(1)
    unlabel_im_info = torch.FloatTensor(1)
    unlabel_num_boxes = torch.LongTensor(1)
    unlabel_gt_boxes = torch.FloatTensor(1)
    unlabel_semi_check = torch.ByteTensor(1)

    # ship to cuda
    if args.cuda:
        unlabel_im_data = unlabel_im_data.cuda()
        unlabel_im_info = unlabel_im_info.cuda()
        unlabel_num_boxes = unlabel_num_boxes.cuda()
        unlabel_gt_boxes = unlabel_gt_boxes.cuda()
        unlabel_semi_check = unlabel_semi_check.cuda()

    # make variable
    unlabel_im_data = Variable(unlabel_im_data)
    unlabel_im_info = Variable(unlabel_im_info)
    unlabel_num_boxes = Variable(unlabel_num_boxes)
    unlabel_gt_boxes = Variable(unlabel_gt_boxes)
    unlabel_semi_check = Variable(unlabel_semi_check)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        model = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        model = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    model.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    #tr_momentum = cfg.TRAIN.MOMENTUM
    #tr_momentum = args.momentum

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        resume_output_dir = os.path.join(args.save_dir, args.arch, args.net, 'pascal_voc_0712_semi')
        load_name = os.path.join(resume_output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        #args.session = checkpoint['session']
        #args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        # optimizer.param_groups[0]['lr'] = args.lr
        # lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        model = nn.DataParallel(model)

    if args.cuda:
        model.cuda()


    # supervised_batch_size = 1
    # unsupervised_batch_size = args.batch_size - supervised_batch_size
    #
    sup_iters_per_epoch = int(train_size / supervised_batch_size)
    unsuper_iters_per_epoch = int(unlabel_train_size / unsupervised_batch_size)

    #ramp_iteration = 10022 * 31
    ramp_iteration = 0

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        dataset.resize_batch()
        unlabel_dataset.resize_batch()
        # setting to train mode
        model.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        unlabel_data_iter = iter(unlabel_dataloader)
        for step in range(sup_iters_per_epoch):
            try:
                sup_data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                sup_data = next(data_iter)

            try:
                upsup_data = next(unlabel_data_iter)
            except StopIteration:
                unlabel_data_iter = iter(unlabel_dataloader)
                upsup_data = next(unlabel_data_iter)


            im_data.data.resize_(sup_data[0].size()).copy_(sup_data[0])
            im_info.data.resize_(sup_data[1].size()).copy_(sup_data[1])
            gt_boxes.data.resize_(sup_data[2].size()).copy_(sup_data[2])
            num_boxes.data.resize_(sup_data[3].size()).copy_(sup_data[3])
            semi_check.data.resize_(sup_data[4].size()).copy_(sup_data[4])

            unlabel_im_data.data.resize_(upsup_data[0].size()).copy_(upsup_data[0])
            unlabel_im_info.data.resize_(upsup_data[1].size()).copy_(upsup_data[1])
            unlabel_gt_boxes.data.resize_(upsup_data[2].size()).copy_(upsup_data[2])
            unlabel_num_boxes.data.resize_(upsup_data[3].size()).copy_(upsup_data[3])
            unlabel_semi_check.data.resize_(upsup_data[4].size()).copy_(upsup_data[4])

            unlabel_im_data_numpy = unlabel_im_data.data.cpu().numpy()
            unlabel_patch_zeros = np.zeros([unsupervised_batch_size, 3, im_data.size()[2], im_data.size()[3]])

            for a in range(unsupervised_batch_size):
                unlabel_patch_zeros[a,:,:,:] = resize(unlabel_im_data_numpy[a,:,:,:], (3, im_data.size()[2], im_data.size()[3]), anti_aliasing=True)
                unlabel_im_info[a,0] = im_data.size()[2]
                unlabel_im_info[a,1] = im_data.size()[3]

            unlabel_im_data = torch.from_numpy(unlabel_patch_zeros).float()
            unlabel_im_data = unlabel_im_data.cuda()
            unlabel_im_data = Variable(unlabel_im_data)

            im_data = torch.cat((im_data,unlabel_im_data),dim=0)
            im_info = torch.cat((im_info,unlabel_im_info),dim=0)
            gt_boxes = torch.cat((gt_boxes,unlabel_gt_boxes),dim=0)
            num_boxes = torch.cat((num_boxes,unlabel_num_boxes),dim=0)
            semi_check = torch.cat((semi_check,unlabel_semi_check),dim=0)


            # print(im_data.size(), unlabel_im_data.size())
            model.zero_grad()
            # print(semi_check)
            rois, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, consistency_loss = model(im_data, im_info, gt_boxes, num_boxes, semi_check, training=True)

            ramp_weight = rampweight(ramp_iteration)
            # print(ramp_weight)
            # consistency_entropy = torch.mul(entropy_loss, ramp_weight)
            # consistency_entropy = torch.mul(consistency_entropy, 5)
            #consistency_entropy = consistency_loss
            consistency_loss = torch.mul(consistency_loss,ramp_weight)

            #semi_rpn_loss_bbox = torch.mul(semi_rpn_loss_bbox, ramp_weight)

            ramp_iteration += 1


            loss = rpn_loss_cls.sum() + rpn_loss_box.sum() \
                   + RCNN_loss_cls.sum() + RCNN_loss_bbox.sum() + consistency_loss.mean() #+ semi_rpn_loss_bbox.mean()
            loss_temp += loss.data[0]
            #
            # rois, cls_prob, bbox_pred, \
            # rpn_loss_cls, rpn_loss_box, \
            # RCNN_loss_cls, RCNN_loss_bbox, \
            # rois_label, consistency_cls, consistency_loc = model(im_data, im_info, gt_boxes, num_boxes, semi_check)
            # model.zero_grad()
            # rois, cls_prob, bbox_pred, \
            # rpn_loss_cls, rpn_loss_box, \
            # RCNN_loss_cls, RCNN_loss_bbox, \
            # rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            #
            # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            #        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            # loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(model, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:       #args.disp_interval
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval     #args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.sum().data[0]
                    loss_rpn_box = rpn_loss_box.sum().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.sum().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.sum().data[0]
                    con_loss = consistency_loss.mean().data[0]
                    # loss_consistency_loc = consistency_loc.mean().data[0]
                    # loss_semi_rpn_loss_bbox = semi_rpn_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    con_loss = consistency_loss.data[0]
                    # loss_consistency_loc = consistency_loc.data[0]
                    # loss_semi_rpn_loss_bbox = semi_rpn_loss_bbox.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, sup_iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\tcon_loss cls: %.7f" \
                      % (con_loss))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
