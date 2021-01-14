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
        self.triplet_loss = [Triplet_Loss() for t in range(3)] ##refine_time
        self.cos_dist =  nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pair_dist = nn.PairwiseDistance(p=2)
        self.distance_layer = distance_layer()
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, triplet_feature, iteration=0, epsilon=1e-10):
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

        ref_score = ref_scores.copy()

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        for r, r_score in enumerate(ref_scores):
            ref_score[r] = F.softmax(r_score, dim=1)
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        max_ind_dict = [0] * len(ref_scores)
        num_refs = len(ref_scores)
        batch_index = [0] * len(final_score_list)
        source_score = [0] * len(ref_scores)

        max_indexes_iou = []

        target_cat = torch.cat((targets[0].get_field('labels').unique(), targets[1].get_field('labels').unique())).tolist()
        duplicate = [int(item) for item, count in collections.Counter(target_cat).items() if count > 1][0]

        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0
            return_loss_dict['loss_triplet%d'%i] = 0

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
                #max_indexes_iou.append(max_index_iou)

                #return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)
            batch_index[idx] = max_ind_dict.copy()

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        triplets = []
        triplet_loss = [0 for j in range(3)] ## refine_time
        close_obj = [] ##refine_time,batch_size

        ### triplet selection
        #duplicate_check = []
        #for b in batch_index:
        #    duplicate_check += list(b[0].keys())
        #duplicate = [item for item, count in collections.Counter(duplicate_check).items() if count > 1][0]
        triplet_batch = [list(x) for x in zip(*batch_index)]

        for r, ref in enumerate(triplet_batch): ## three == refine_time
            close_batch = []
            box_per_batch = proposals[0].bbox.shape[0]

            b1_triplet_feature = triplet_feature[:box_per_batch]
            b2_triplet_feature = triplet_feature[box_per_batch:]
            b1_ref_score = ref_score[r][:box_per_batch]
            b2_ref_score = ref_score[r][box_per_batch:]

            #a = torch.tensor(ref[0].get(duplicate))
            #p = torch.tensor(ref[1].get(duplicate))
            #n = source_score[r][:,0].argmax()

            if len(ref[1]) == 0 or len(ref[0]) == 0:
                import IPython; IPython.embed()
            if len(ref[0]) > len(ref[1]):
                ref[1] = ref[1] * divmod(len(ref[0]),len(ref[1]))[0] + ref[1][:divmod(len(ref[0]),len(ref[1]))[1]]
            elif len(ref[0]) < len(ref[1]):
                ref[0] = ref[0] * divmod(len(ref[1]),len(ref[0]))[0] + ref[0][:divmod(len(ref[1]),len(ref[0]))[1]]

            a = torch.tensor(ref[0])
            p = torch.tensor(ref[1])
            #import IPython; IPython.embed()
            a_feat = triplet_feature[:box_per_batch][a].squeeze(1)
            p_feat = triplet_feature[box_per_batch:][p].squeeze(1)

            #while a_feat.shape[0] == p_feat.shape[0]:
            #if a_feat.shape[0] < p_feat.shape[0]:
            #    a_feat = a_feat.repeat(p_feat.shape[0],1)
            #elif a_feat.shape[0] > p_feat.shape[0]:
            #    p_feat = p_feat.repeat(a_feat.shape[0],1)

            b1_n = b1_ref_score[:,0].topk(a_feat.shape[0])[1]
            b2_n = b2_ref_score[:,0].topk(a_feat.shape[0])[1]

            b1_n_feat = triplet_feature[b1_n].squeeze(1)
            b2_n_feat = triplet_feature[b2_n].squeeze(1)

            triplet_loss[r] += self.triplet_loss[r](a_feat, p_feat, b1_n_feat)
            triplet_loss[r] += self.triplet_loss[r](a_feat, p_feat, b2_n_feat)

            ## TODO triplet loss by batch?

            b1_dist = torch.zeros(b1_triplet_feature.shape[0], dtype=torch.float, device=device)
            b2_dist = torch.zeros(b2_triplet_feature.shape[0], dtype=torch.float, device=device)
            for i in range(len(a)):
                #b1_dist += self.cos_dist(b1_triplet_feature, a_feat[i].unsqueeze(0))
                #b2_dist += self.cos_dist(b2_triplet_feature, p_feat[i].unsqueeze(0))
                b1_dist = self.pair_dist(b1_triplet_feature, a_feat[i].unsqueeze(0))
                b2_dist = self.pair_dist(b2_triplet_feature, p_feat[i].unsqueeze(0))

            #b1_dist = self.cos_dist(triplet_feature, a_feat)
            #b2_dist = self.cos_dist(triplet_feature, p_feat)

            #mean_dist = (self.cos_dist(triplet_feature, a_feat) + self.cos_dist(triplet_feature, p_feat))/2
            b1_dist = b1_dist/len(a)
            b2_dist = b2_dist/len(a)
            #mean_dist = (b1_dist + b2_dist)/2

            #close_ind = (mean_dist > 0.5).nonzero()
            #b1_close_1 = close_ind[:(close_ind < box_per_batch).nonzero().shape[0]]
            #b2_close_2 = close_ind[(close_ind < box_per_batch).nonzero().shape[0]:] - proposals[0].bbox.shape[0]


            #b1_close = (b1_dist > 0.5).nonzero(as_tuple=False)
            #b2_close = (b2_dist > 0.5).nonzero(as_tuple=False)

            b1_close = (b1_dist < b1_dist[a].mean()).nonzero(as_tuple=False)
            b2_close = (b2_dist < b2_dist[p].mean()).nonzero(as_tuple=False)
            #import IPython; IPython.embed()
            #b1_close = b1_close[:(b1_close < box_per_batch).nonzero().shape[0]]
            #b2_close = b1_close[:(b1_close < box_per_batch).nonzero().shape[0]] - proposals[0].bbox.shape[0]


            close_batch.append([e.item() for e in b1_close])
            close_batch.append([e.item() for e in b2_close])

            close_obj.append(close_batch)

            return_loss_dict['loss_triplet%d'%r] = triplet_loss[r] / 2

        ### find more objects ###
        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)

            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights = self.distance_layer(proposals_per_image, source_score, labels_per_im, device, close_obj[i][idx], duplicate)
                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)
        ### triplet end ###
        if (iteration == 3000):
            import IPython; IPython.embed()
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
