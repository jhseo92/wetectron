# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch.nn import functional as F
import collections

from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async
from wetectron.structures.bounding_box import BoxList, BatchBoxList
from wetectron.structures.boxlist_ops import boxlist_nms
from wetectron.modeling.matcher import Matcher
from wetectron.modeling.roi_heads.weak_head.gcn import GCNLayer, SkipConnection, GatedSkipConnection, GCNBlock, ReadOut, Predictor, GCNNet

from .pseudo_label_generator import oicr_layer, mist_layer


def generate_img_label(num_classes, labels, device):
    img_label = torch.zeros(num_classes)
    img_label[labels.long()] = 1
    img_label[0] = 0
    return img_label.to(device)


def compute_avg_img_accuracy(labels_per_im, score_per_im, num_classes):
    """
       the accuracy of top-k prediction
       where the k is the number of gt classes
    """
    num_pos_cls = max(labels_per_im.sum().int().item(), 1)
    cls_preds = score_per_im.topk(num_pos_cls)[1]
    accuracy_img = labels_per_im[cls_preds].mean()
    return accuracy_img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def get_geometry(bbox, score, device, p):
    centroids = torch.zeros((0,2), dtype=float, device=device)
    box_info = torch.zeros((0,4), dtype=float, device=device)

    for b in bbox.bbox:
        x_centre = (b[0] + b[2]) / 2
        y_centre = (b[1] + b[3]) / 2
        w = (b[2] - b[0]) / bbox.size[0]
        h = (b[3] - b[1]) / bbox.size[1]
        box_size = (w*h) / (bbox.size[0] *bbox.size[1])
        centre = torch.tensor([x_centre, y_centre]).to(device).unsqueeze(0)
        centroids = torch.cat((centroids, centre))
        box_info = torch.cat((box_info, torch.tensor([w,h,w/h,box_size]).to(device).unsqueeze(0)))
    ### normalize ###
    centroids[:,0] = centroids[:,0]/bbox.size[0]
    centroids[:,1] = centroids[:,1]/bbox.size[1]
    ### normalize ###
    centroids_trans = centroids - centroids[score[:, p].argmax()]
    mean_vector = centroids - centroids[score[:, p].argmax()]
    distances = torch.cdist(centroids, centroids[score[:, p].argmax()].unsqueeze(0), p=2).to(device)
    #import IPython; IPython.embed()
    return centroids, centroids_trans, mean_vector, distances, box_info

@registry.ROI_WEAK_LOSS.register("WSDDNLoss")
class WSDDNLossComputation(object):
    """ Computes the loss for WSDDN."""
    def __init__(self, cfg):
        self.type = "WSDDN"

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-10):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])

        Returns:
            img_loss (Tensor)
            accuracy_img (Tensor): the accuracy of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        class_score_list = class_score.split([len(p) for p in proposals])
        det_score_list = final_det_score.split([len(p) for p in proposals])

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        total_loss = 0
        accuracy_img = 0
        for idx, (final_score_per_im, cls_score_per_im, det_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, class_score_list, det_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            _labels = labels_per_im[1:]
            positive_classes = torch.arange(_labels.shape[0])[_labels==1].to(device).add(1)

            cls_feature = torch.zeros((len(proposals_per_image), 3*21), dtype=torch.float, device=device)
            det_feature = torch.zeros((len(proposals_per_image), 0), dtype=torch.float, device=device)

            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            ### img_geometry ###
            for p in positive_classes:
                _, _, mean_vector, distances, box_info = get_geometry(proposals_per_image, det_score_per_im, device, p)
                # 2, 2, 1, 4
                cls_feature[:,p*3:(p+1)*3] = torch.cat((mean_vector, distances), dim=1)
                #cls_feature = torch.cat((cls_feature, mean_vector, distances), dim=1)
            det_feature = torch.cat((det_feature, box_info), dim=1)

            adj_iou = boxlist_iou(proposals_per_image, proposals_per_image)
            cls_feature = torch.cat( (cls_feature, cls_score_per_im), dim=1).float()
            det_feature = torch.cat( (det_feature, det_score_per_im), dim=1).float()

            cls_net = GCNNet(1, 1, cls_feature.shape[-1], cls_feature.shape[-1], 1, cls_feature.shape[-1], cls_feature.shape[-1], cls_feature.shape[-1], 21, False, 'gsc').to('cuda')
            det_net = GCNNet(1, 1, det_feature.shape[-1], det_feature.shape[-1], 1, det_feature.shape[-1], det_feature.shape[-1], det_feature.shape[-1], 21, False, 'gsc').to('cuda')
            ### n_block, n_layer, in_dim, hidden_dim, n_atom, pred_dim1, pred_dim2, pred_dim3
            cls_feature = cls_net(cls_feature, adj_iou)
            det_feature = det_net(det_feature, adj_iou)
            cls_score = F.softmax(cls_feature, dim=1)
            det_score = F.softmax(det_feature, dim=0)

            gcn_img_score_per_im = torch.clamp(torch.sum(cls_score * det_score, dim=0), min=epsilon, max=1-epsilon)

            ### img_geometry ###
            #img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            #return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            total_loss += F.binary_cross_entropy(gcn_img_score_per_im, labels_per_im.clamp(0, 1))
            accuracy_img += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)

        '''class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        total_loss = 0
        accuracy_img = 0
        for final_score_per_im, targets_per_im in zip(final_score_list, targets):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            total_loss += F.binary_cross_entropy(img_score_per_im, labels_per_im)
            accuracy_img += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)

        total_loss = total_loss / len(final_score_list)
        accuracy_img = accuracy_img / len(final_score_list)
'''
        return dict(loss_img=total_loss), dict(accuracy_img=accuracy_img)


@registry.ROI_WEAK_LOSS.register("RoILoss")
class RoILossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.type = "RoI_loss"
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')
        #self.gcn = GCNNet(3, 3, 42, 4096, 1, 4096, 2048, 1024, 20, False, 'gsc').to('cuda')
    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-10):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
            ref_scores
            proposals
            targets
        Returns:
            return_loss_dict (dictionary): all the losses
            return_acc_dict (dictionary): all the accuracies of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        class_score_list = class_score.split([len(p) for p in proposals])
        det_score_list = final_det_score.split([len(p) for p in proposals])

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)
        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, cls_score_per_im, det_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, class_score_list, det_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            _labels = labels_per_im[1:]
            positive_classes = torch.arange(_labels.shape[0])[_labels==1].to(device).add(1)

            cls_feature = torch.zeros((len(proposals_per_image), 3*21), dtype=torch.float, device=device)
            det_feature = torch.zeros((len(proposals_per_image), 0), dtype=torch.float, device=device)

            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            ### img_geometry ###
            for p in positive_classes:
                _, _, mean_vector, distances, box_info = get_geometry(proposals_per_image, det_score_per_im, device, p)
                # 2, 2, 1, 4
                cls_feature[:,p*3:(p+1)*3] = torch.cat((mean_vector, distances), dim=1)
                #cls_feature = torch.cat((cls_feature, mean_vector, distances), dim=1)
            det_feature = torch.cat((det_feature, box_info), dim=1)

            adj_iou = boxlist_iou(proposals_per_image, proposals_per_image)
            cls_feature = torch.cat( (cls_feature, cls_score_per_im), dim=1).float()
            det_feature = torch.cat( (det_feature, det_score_per_im), dim=1).float()

            cls_net = GCNNet(1, 1, cls_feature.shape[-1], 4096, 1, 4096, 4096, 4096, 21, False, 'gsc').to('cuda')
            det_net = GCNNet(1, 1, det_feature.shape[-1], 4096, 1, 4096, 4096, 4096, 21, False, 'gsc').to('cuda')
            ### n_block, n_layer, in_dim, hidden_dim, n_atom, pred_dim1, pred_dim2, pred_dim3
            cls_feature = cls_net(cls_feature, adj_iou)
            det_feature = det_net(det_feature, adj_iou)
            cls_score = F.softmax(cls_feature, dim=1)
            det_score = F.softmax(det_feature, dim=0)

            gcn_img_score_per_im = torch.clamp(torch.sum(cls_score * det_score, dim=0), min=epsilon, max=1-epsilon)

            ### img_geometry ###
            #img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            #return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            return_loss_dict['loss_img'] += F.binary_cross_entropy(gcn_img_score_per_im, labels_per_im.clamp(0, 1))

            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights = self.roi_layer(proposals_per_image, source_score, labels_per_im, device)
                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)
                ### positive_classes = c.add(1)
                '''for p in positive_classes:
                    if (pseudo_labels == p).nonzero(as_tuple=False).shape[0] == 1:
                        p_gt = (pseudo_labels == p).nonzero(as_tuple=False).squeeze(0)
                    else :
                        p_gt = (pseudo_labels == p).nonzero(as_tuple=False).squeeze()
                    gt_box = BoxList(targets_per_im.bbox[targets_per_im.get_field('labels') == p], proposals_per_image.size, mode=proposals_per_image.mode)
                    p_box = BoxList(proposals_per_image.bbox[p_gt], proposals_per_image.size, mode=proposals_per_image.mode)
                    p_box.add_field('scores', source_score[p_gt,p])
                    p_score = p_box.get_field('scores')
                    gt_iou = boxlist_iou(gt_box, p_box)
                    box_iou = boxlist_iou(p_box, p_box)
                    box_iou_mean = box_iou.mean(dim=1)
                    centroids, centroids_trans, mean_vector, distances = get_geometry(p_box.bbox, p_score, device)
                    import IPython; IPython.embed()'''
            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            return_loss_dict[l] /= len(final_score_list)
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


@registry.ROI_WEAK_LOSS.register("RoIRegLoss")
class RoIRegLossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')
        # for regression
        self.cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        # for partial labels
        self.roi_refine = cfg.MODEL.ROI_WEAK_HEAD.ROI_LOSS_REFINE
        self.partial_label = cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS
        assert self.partial_label in ['none', 'point', 'scribble']
        self.proposal_scribble_matcher = Matcher(
            0.5, 0.5, allow_low_quality_matches=False,
        )

    def filter_pseudo_labels(self, pseudo_labels, proposal, target):
        """ refine pseudo labels according to partial labels """
        if 'scribble' in target.fields() and self.partial_label=='scribble':
            scribble = target.get_field('scribble')
            match_quality_matrix_async = boxlist_iou_async(scribble, proposal)
            _matched_idxs = self.proposal_scribble_matcher(match_quality_matrix_async)
            pseudo_labels[_matched_idxs < 0] = 0
            matched_idxs = _matched_idxs.clone().clamp(0)
            _labels = target.get_field('labels')[matched_idxs]
            pseudo_labels[pseudo_labels != _labels.long()] = 0

        elif 'click' in target.fields() and self.partial_label=='point':
            clicks = target.get_field('click').keypoints
            clicks_tiled = torch.unsqueeze(torch.cat((clicks, clicks), dim=1), dim=1)
            num_obj = clicks.shape[0]
            box_repeat = torch.cat([proposal.bbox.unsqueeze(0) for _ in range(num_obj)], dim=0)
            diff = clicks_tiled - box_repeat
            matched_ids = (diff[:,:,0] > 0) * (diff[:,:,1] > 0) * (diff[:,:,2] < 0) * (diff[:,:,3] < 0)
            matched_cls = matched_ids.float() * target.get_field('labels').view(-1, 1)
            pseudo_labels_repeat = torch.cat([pseudo_labels.unsqueeze(0) for _ in range(matched_ids.shape[0])])
            correct_idx = (matched_cls == pseudo_labels_repeat.float()).sum(0)
            pseudo_labels[correct_idx==0] = 0

        return pseudo_labels

    def __call__(self, class_score, det_score, ref_scores, ref_bbox_preds, proposals, targets, epsilon=1e-10):
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]
        ref_bbox_preds = [rbp.split([len(p) for p in proposals]) for rbp in ref_bbox_preds]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)
        for i in range(num_refs):
            return_loss_dict['loss_ref_cls%d'%i] = 0
            return_loss_dict['loss_ref_reg%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):

            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))

            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                pseudo_labels, loss_weights, regression_targets = self.roi_layer(
                    proposals_per_image, source_score, labels_per_im, device, return_targets=True
                )
                if self.roi_refine:
                    pseudo_labels = self.filter_pseudo_labels(pseudo_labels, proposals_per_image, targets_per_im)

                lmda = 3 if i == 0 else 1
                # classification
                return_loss_dict['loss_ref_cls%d'%i] += lmda * torch.mean(
                    F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights
                )
                # regression
                sampled_pos_inds_subset = torch.nonzero(pseudo_labels>0, as_tuple=False).squeeze(1)
                labels_pos = pseudo_labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

                box_regression = ref_bbox_preds[i][idx]
                reg_loss = lmda * torch.sum(smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    beta=1, reduction=False) * loss_weights[sampled_pos_inds_subset, None]
                )
                reg_loss /= pseudo_labels.numel()
                return_loss_dict['loss_ref_reg%d'%i] += reg_loss

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            return_loss_dict[l] /= len(final_score_list)
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


def make_roi_weak_loss_evaluator(cfg):
    func = registry.ROI_WEAK_LOSS[cfg.MODEL.ROI_WEAK_HEAD.LOSS]
    return func(cfg)
