import torch
import torch.nn as nn
import torchvision

from utils import *

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training \
            else model_config['rpn_test_prenms_topk']
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]
        
        # For the vgg16 case stride would be 16 for both h and w
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h

        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
        
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)

        return anchors
    
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        
        # Get (gt_boxes, num_anchors_in_image) IOU matrix
        iou_matrix = get_iou(gt_boxes, anchors)
        
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
        
        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2
        
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        
        # Get all the anchors indexes to update
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Set all foreground anchor labels as 1
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        
        # Set all background anchor labels as 0
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        
        # Set all to be ignored anchor labels as -1
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        # Pre NMS Filtering
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))
        
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # Small boxes based on width and height filtering
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        
        # NMS based on objectness scores
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torchvision.ops.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        # Sort by objectness
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        # Post NMS topk filtering
        proposals, cls_scores = (proposals[post_nms_keep_indices[:self.rpn_topk]],
                                 cls_scores[post_nms_keep_indices[:self.rpn_topk]])
        
        return proposals, cls_scores
    
    def forward(self, image, feat, target=None):
        r"""
        Main method for RPN does the following:
        1. Call RPN specific conv layers to generate classification and
            bbox transformation predictions for anchors
        2. Generate anchors for entire image
        3. Transform generated anchors based on predicted bbox transformation to generate proposals
        4. Filter proposals
        5. For training additionally we do the following:
            a. Assign target ground truth labels and boxes to each anchors
            b. Sample positive and negative anchors
            c. Compute classification loss using sampled pos/neg anchors
            d. Compute Localization loss using sampled pos anchors
        """
        # Call RPN layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # Generate anchors
        anchors = self.generate_anchors(image, feat)
        
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1])
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        
        proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred.detach().reshape(-1, 1, 4), anchors)
        proposals = proposals.reshape(proposals.size(0), 4)
        
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }
        if not self.training or target is None:
            # If we are not training no need to do anything
            return rpn_output
        else:
            # Assign gt box and label for each anchor
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0])
            
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=self.rpn_pos_count,
                total_count=self.rpn_batch_size)
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            
            localization_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        box_transform_pred[sampled_pos_idx_mask],
                        regression_targets[sampled_pos_idx_mask],
                        beta=1 / 9,
                        reduction="sum",
                    )
                    / (sampled_idxs.numel())
            ) 

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_scores[sampled_idxs].flatten(),
                                                                            labels_for_anchors[sampled_idxs].flatten())

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output