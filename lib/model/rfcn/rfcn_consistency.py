import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.psroi_pooling.modules.psroi_pool import PSRoIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class _RFCN(nn.Module):
    """ R-FCN """
    def __init__(self, classes, class_agnostic):
        super(_RFCN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.box_num_classes = 1 if class_agnostic else self.n_classes

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_psroi_pool_cls = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.n_classes)
        self.RCNN_psroi_pool_loc = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.box_num_classes * 4)
        self.pooling = nn.AvgPool2d(kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):
        # classification loss
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        return RCNN_loss_cls, RCNN_loss_bbox

    def ohem_detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

        def log_sum_exp(x):
            x_max = x.data.max()
            return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

        num_hard = cfg.TRAIN.BATCH_SIZE * self.batch_size
        pos_idx = rois_label > 0
        num_pos = pos_idx.int().sum()

        # classification loss
        num_classes = cls_score.size(1)
        weight = cls_score.data.new(num_classes).fill_(1.)
        weight[0] = num_pos.data[0] / num_hard

        conf_p = cls_score.detach()
        conf_t = rois_label.detach()

        # rank on cross_entropy loss
        loss_c = log_sum_exp(conf_p) - conf_p.gather(1, conf_t.view(-1,1))
        loss_c[pos_idx] = 100. # include all positive samples
        _, topk_idx = torch.topk(loss_c.view(-1), num_hard)
        loss_cls = F.cross_entropy(cls_score[topk_idx], rois_label[topk_idx], weight=weight)

        # bounding box regression L1 loss
        pos_idx = pos_idx.unsqueeze(1).expand_as(bbox_pred)
        loc_p = bbox_pred[pos_idx].view(-1, 4)
        loc_t = rois_target[pos_idx].view(-1, 4)
        loss_box = F.smooth_l1_loss(loc_p, loc_t)

        return loss_cls, loss_box

    # def forward(self, im_data, im_info, gt_boxes, num_boxes):
    def forward(self, im_data, im_info, gt_boxes, num_boxes, semi_check, training=False):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        semi_check = semi_check.data
        self.batch_size = im_data.size(0)

        im_data_flip = im_data.clone()
        im_data_flip = flip(im_data_flip,3)


        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        base_feat_flip = self.RCNN_base(im_data_flip)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)


        consistency_rois = rois.clone()

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training and int(semi_check):
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = Variable(torch.cuda.FloatTensor([0]))
            # rois_label = None
            rois_target = Variable(torch.cuda.FloatTensor([0]))
            rois_inside_ws = Variable(torch.cuda.FloatTensor([0]))
            rois_outside_ws = Variable(torch.cuda.FloatTensor([0]))
            rpn_loss_cls = Variable(torch.cuda.FloatTensor([0]))
            rpn_loss_bbox = Variable(torch.cuda.FloatTensor([0]))

        rois = Variable(rois)
        consistency_rois = Variable(consistency_rois)

        base_feat = self.RCNN_conv_new(base_feat)

        # do roi pooling based on predicted rois
        cls_feat = self.RCNN_cls_base(base_feat)
        pooled_feat_cls = self.RCNN_psroi_pool_cls(cls_feat, rois.view(-1, 5))
        cls_score = self.pooling(pooled_feat_cls)
        cls_score = cls_score.squeeze()

        bbox_base = self.RCNN_bbox_base(base_feat)
        pooled_feat_loc = self.RCNN_psroi_pool_loc(bbox_base, rois.view(-1, 5))
        pooled_feat_loc = self.pooling(pooled_feat_loc)
        bbox_pred = pooled_feat_loc.squeeze()

        if self.training and not self.class_agnostic and int(semi_check):
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = Variable(torch.cuda.FloatTensor([0]))
        RCNN_loss_bbox = Variable(torch.cuda.FloatTensor([0]))

        if self.training and int(semi_check):
            loss_func = self.ohem_detect_loss if cfg.TRAIN.OHEM else self.detect_loss
            RCNN_loss_cls, RCNN_loss_bbox = loss_func(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)




        # # unlabeled loss

        consistency_feat = self.RCNN_cls_base(base_feat)
        consistency_feat_cls = self.RCNN_psroi_pool_cls(consistency_feat, consistency_rois.view(-1, 5))
        consistency_cls_score = self.pooling(consistency_feat_cls)
        consistency_cls_score = consistency_cls_score.squeeze()
        consistency_cls_prob = F.softmax(consistency_cls_score, dim=1)

        base_feat_flip = self.RCNN_conv_new(base_feat_flip)
        consistency_feat_flip = self.RCNN_cls_base(base_feat_flip)

        consistency_rois_flip = consistency_rois.clone()

        consistency_rois_flip[:, :, 1] = im_info[0][1] - (consistency_rois.clone()[:, :, 3] + 1)
        consistency_rois_flip[:,:,3] = im_info[0][1] - (consistency_rois.clone()[:,:,1] + 1 )

        consistency_feat_cls_flip = self.RCNN_psroi_pool_cls(consistency_feat_flip, consistency_rois_flip.view(-1, 5))
        consistency_cls_score_flip = self.pooling(consistency_feat_cls_flip)
        consistency_cls_score_flip = consistency_cls_score_flip.squeeze()
        consistency_cls_prob_flip = F.softmax(consistency_cls_score_flip, dim=1)

        consistency_bbox_feat = self.RCNN_bbox_base(base_feat)
        consistency_bbox_feat_flip = self.RCNN_bbox_base(base_feat_flip)
        consistency_bbox_feat_pool = self.RCNN_psroi_pool_loc(consistency_bbox_feat, consistency_rois.view(-1, 5))
        consistency_bbox_feat_pool_flip = self.RCNN_psroi_pool_loc(consistency_bbox_feat_flip, consistency_rois_flip.view(-1, 5))
        consistency_bbox_feat_pool = self.pooling(consistency_bbox_feat_pool)
        consistency_bbox_feat_pool_flip = self.pooling(consistency_bbox_feat_pool_flip)
        consistency_bbox_pred = consistency_bbox_feat_pool.squeeze()
        consistency_bbox_pred_flip = consistency_bbox_feat_pool_flip.squeeze()

        consistency_bbox_pred_view = consistency_bbox_pred.view(consistency_bbox_pred.size(0), int(consistency_bbox_pred.size(1) / 4), 4)
        consistency_bbox_pred_view_flip = consistency_bbox_pred_flip.view(consistency_bbox_pred_flip.size(0),
                                                                int(consistency_bbox_pred_flip.size(1) / 4), 4)

        consistency_bbox_pred_view_flip = torch.mul(consistency_bbox_pred_view_flip, 1000)
        consistency_bbox_pred_view_flip[:, :, 0] = torch.mul(consistency_bbox_pred_view_flip[:, :, 0], -1)
        consistency_bbox_pred_view_flip = torch.div(consistency_bbox_pred_view_flip, 1000)

        # consistency_loc = torch.mean(torch.pow(bbox_pred_view_unflip - bbox_pred_view_flip, exponent=2))

        conf_class = consistency_cls_prob[:,1:].clone()
        background_score = consistency_cls_prob[:,0].clone()
        each_val, each_index = torch.max(conf_class,dim=1)
        mask_val = each_val > background_score
        mask_val = mask_val.data

        conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

        if (mask_val.sum() > 0):
            mask_conf_index = mask_val.unsqueeze(1).expand_as(consistency_cls_prob)
            conf_mask_sample = consistency_cls_prob.clone()
            conf_sample = conf_mask_sample[mask_conf_index].view(-1, 21)

            conf_mask_sample_flip = consistency_cls_prob_flip.clone()
            conf_sample_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 21)

            conf_sampled = conf_sample + 1e-7
            conf_sampled_flip = conf_sample_flip + 1e-7
            consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(), conf_sampled_flip.detach()).sum(
                -1).mean()
            consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(), conf_sampled.detach()).sum(
                -1).mean()
            consistency_cls = torch.div(consistency_conf_loss_a + consistency_conf_loss_b, 2)

            mask_loc_index = mask_val.unsqueeze(1).unsqueeze(2).expand_as(consistency_bbox_pred_view)
            loc_mask_sample = consistency_bbox_pred_view.clone()
            loc_sample = loc_mask_sample[mask_loc_index].view(-1, 4)

            loc_mask_sample_flip = consistency_bbox_pred_view_flip.clone()
            loc_sample_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

            consistency_loc = torch.mean(torch.pow(loc_sample - loc_sample_flip, exponent=2))

            consistency_loss = consistency_cls + consistency_loc

        else:
            consistency_loss = Variable(torch.cuda.FloatTensor([0]))


        return consistency_rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, consistency_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_conv_1x1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_base, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
