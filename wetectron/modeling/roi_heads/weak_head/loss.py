# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
import collections
import torch.nn as nn
from torch.nn import functional as F

from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async
from wetectron.modeling.matcher import Matcher
from wetectron.modeling.roi_heads.triplet_head.triplet_loss import Triplet_Loss

from .pseudo_label_generator import oicr_layer, mist_layer, distance_layer


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

        return dict(loss_img=total_loss), dict(accuracy_img=accuracy_img)


@registry.ROI_WEAK_LOSS.register("RoILoss")
class RoILossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.type = "RoI_loss"
        self.triplet_loss = Triplet_Loss() ##refine_time
        self.cos_dist =  nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.distance_layer = distance_layer()
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, triplet_feature, iteration =0 , epsilon=1e-10):
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

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_score = ref_scores.copy()
        logits_score = [0] *2

        for r, r_score in enumerate(ref_scores):
            ref_score[r] = F.softmax(r_score, dim=1)
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        for r, r_score in enumerate(ref_scores):
            for b, b_r_score in enumerate(r_score):
                logits_score[b] += b_r_score
        logits_score[0] = logits_score[0]/3
        logits_score[1] = logits_score[1]/3

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        max_ind_dict = [0] * len(ref_scores)
        num_refs = len(ref_scores)
        batch_index = [0] * len(final_score_list)
        source_score = [0] * len(ref_scores)
        return_loss_dict['loss_triplet'] = 0

        target_cat = torch.cat((targets[0].get_field('labels').unique(), targets[1].get_field('labels').unique())).tolist()
        duplicate = [int(item) for item, count in collections.Counter(target_cat).items() if count > 1]

        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)

            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score[i] = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights, max_index = self.roi_layer(proposals_per_image, source_score[i], labels_per_im, device, duplicate)
                                                        ### pseudo_label_generator
                max_ind_dict[i] = max_index

                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)
            batch_index[idx] = max_ind_dict.copy()

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        triplets = []
        triplet_loss = [0] ## refine_time
        close_obj = [] ##refine_time,batch_size

        ### triplet selection
        triplet_batch = [list(x) for x in zip(*batch_index)]

        box_per_batch = proposals[0].bbox.shape[0]
        tot_score = torch.zeros_like(ref_score[0], dtype=float, device=device)

        for r_score in ref_score:
            tot_score += r_score
        avg_score = tot_score/len(source_score)

        b1_avg_score = avg_score[:box_per_batch]
        b2_avg_score = avg_score[box_per_batch:]

        b1_triplet_feature = triplet_feature[:box_per_batch]
        b2_triplet_feature = triplet_feature[box_per_batch:]

        b1_max_index = b1_avg_score[:,duplicate[0]].argmax()
        b2_max_index = b2_avg_score[:,duplicate[0]].argmax()
        neg_index = b2_avg_score[:,0].argmax()

        average_score = []
        average_score.append(b1_avg_score)
        average_score.append(b2_avg_score)
        return_loss_dict['loss_sampling'] = 0


        a_feature = b1_triplet_feature[b1_max_index].unsqueeze(0)
        p_feature = b2_triplet_feature[b2_max_index].unsqueeze(0)
        n_feature = b2_triplet_feature[neg_index].unsqueeze(0)

        avg_max_ind = [0] * len(final_score_list)
        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            pseudo_labels_mining, loss_weights_mining, max_index = self.roi_layer(proposals_per_image, average_score[idx], labels_per_im, device, duplicate)
            avg_max_ind[idx] = max_index

        a_feat = b1_triplet_feature[torch.tensor(avg_max_ind[0][duplicate[0]])].squeeze(1)
        p_feat = b2_triplet_feature[torch.tensor(avg_max_ind[1][duplicate[0]])].squeeze(1)
        try:
            if a_feat.shape[0] <= p_feat.shape[0]:
                p_feat = p_feat[:a_feat.shape[0],:]
            else:
                a_feat = a_feat[:p_feat.shape[0],:]
        except:
            import IPython; IPython.embed()

        n_feat = b2_triplet_feature[b2_avg_score[:,0].topk(a_feat.shape[0])[1]].squeeze()
        triplet_loss = self.triplet_loss(a_feature, p_feature, n_feature) / a_feat.shape[0]

        b1_dist = torch.zeros(triplet_feature.shape[0], dtype=torch.float, device=device)
        b2_dist = torch.zeros(triplet_feature.shape[0], dtype=torch.float, device=device)

        for i in range(a_feat.shape[0]):
            b1_dist += self.cos_dist(triplet_feature, a_feat[i].unsqueeze(0))
            b2_dist += self.cos_dist(triplet_feature, p_feat[i].unsqueeze(0))

        b1_dist = b1_dist / a_feat.shape[0]
        b2_dist = b2_dist / p_feat.shape[0]
        mean_dist = (b1_dist + b2_dist)/2
        '''try:
            b1_dist = self.cos_dist(triplet_feature, a_feature)
            b2_dist = self.cos_dist(triplet_feature, p_feature)
            #b1_dist = self.l2_dist(triplet_feature, a_feature)
            #b2_dist = self.l2_dist(triplet_feature, p_feature)
        except:
            import IPython; IPython.embed()
        '''
        #mean_dist = (b1_dist + b2_dist)/2
        close_ind = (mean_dist >  0.5).nonzero(as_tuple=False)


        b1_close = close_ind[:(close_ind < box_per_batch).nonzero(as_tuple=False).shape[0]]
        b2_close = close_ind[(close_ind < box_per_batch).nonzero(as_tuple=False).shape[0]:] - proposals[0].bbox.shape[0]
        close_obj.append(b1_close)
        close_obj.append(b2_close)
        return_loss_dict['loss_triplet'] = mean_dist.max().item() * triplet_loss

        ### find more objects ###
        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)

            pseudo_labels_mining, loss_weights_mining = self.distance_layer(proposals_per_image, average_score[idx], labels_per_im, device, close_obj[idx], duplicate)

            return_loss_dict['loss_sampling'] += lmda * torch.mean(F.cross_entropy(logits_score[idx], pseudo_labels_mining, reduction='none') * loss_weights_mining)

        ### triplet end ###
        assert len(final_score_list) != 0

        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            if l == 'loss_triplet':
                return_loss_dict[l] = return_loss_dict[l]
            else:
                return_loss_dict[l] /= len(final_score_list)
                return_acc_dict[a] /= len(final_score_list)
        ### TODO triplet_loss diviced by batch size?

        if iteration == 5000 :
            import IPython; IPython.embed()

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
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1)).item()

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
