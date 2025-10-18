"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple

import torch
torch._dynamo.config.suppress_errors = True

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.losses.criterion import SetCriterion
from .modeling.losses.one2many_criterion import One2ManySetCriterion
from .modeling.losses.matcher import HungarianMatcher
from .modeling.losses.one2many_matcher import HungarianOne2ManyMatcher
from .modeling.transformer_decoder.box_regression import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits
VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        use_sigomid: bool,
        # FC-CLIP
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        use_one2many_head: bool
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.use_sigomid = use_sigomid

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask
        self.use_one2many_head = use_one2many_head

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    @torch.compiler.disable(recursive=True)
    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Whether to use one2many head
        use_one2many_head = cfg.MODEL.ZEG_FC.USE_ONE2MANY_HEAD

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        class_focal_weight = cfg.MODEL.MASK_FORMER.CLASS_FOCAL_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        sem_dice_weight = cfg.MODEL.MASK_FORMER.SEM_DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sem_mask_weight = cfg.MODEL.MASK_FORMER.SEM_MASK_WEIGHT
        bbox_weight = cfg.MODEL.MASK_FORMER.BBOX_WEIGHT
        giou_weight = cfg.MODEL.MASK_FORMER.GIOU_WEIGHT
        round_weight = cfg.MODEL.MASK_FORMER.ROUDN_WEIGHT

        # building criterion
        matcher_class = HungarianOne2ManyMatcher if use_one2many_head else HungarianMatcher
        matcher = matcher_class(
            cost_class=class_weight,
            cost_objectness=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            use_nel_cost=cfg.MODEL.FC_CLIP.USE_NEL_COST,
            focal_alpha=cfg.MODEL.FC_CLIP.FOCAL_ALPHA,
            focal_gamma=cfg.MODEL.FC_CLIP.FOCAL_GAMMA
        )

        weight_dict = {
            "loss_ce": class_weight, 
            "loss_objectness": class_weight,
            "loss_label_focal": class_focal_weight,
            "loss_mask": mask_weight, 
            "loss_semantic_mask": sem_mask_weight,
            "loss_dice": dice_weight,
            "loss_semantic_dice": sem_dice_weight,
            "loss_round": round_weight,
            "loss_bbox": bbox_weight,
            "loss_giou": giou_weight
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["objectness", "masks", "semantic", "boxes"]
        if use_one2many_head:
            losses.append("rounds")

        criterion_class = One2ManySetCriterion if use_one2many_head else SetCriterion
        criterion = criterion_class(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            use_nel_loss=cfg.MODEL.FC_CLIP.USE_NEL_COST,
            focal_alpha=cfg.MODEL.FC_CLIP.FOCAL_ALPHA,
            focal_gamma=cfg.MODEL.FC_CLIP.FOCAL_GAMMA
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "use_one2many_head": cfg.MODEL.ZEG_FC.USE_ONE2MANY_HEAD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            "use_sigomid": cfg.MODEL.FC_CLIP.USE_NEL_COST,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_obj_results = outputs.get("pred_objectness")
            mask_cls_results = outputs.get("pred_logits")
            mask_pred_results = outputs.get("pred_panoptic_masks")
            mask_sem_results = outputs.get("pred_semantic_masks")
            mask_box_results = outputs.get("pred_boxes")
            mask_round_results = outputs.get("pred_round")

            if False:
                clip_feature = features["clip_vis_dense"]  # [B, D_clip_raw, Hc, Wc]

                # Project dense features into CLIP embedding space (same head used for pooled features)
                if "convnext" in self.backbone.model_name.lower():
                    dense_proj = self.backbone.visual_prediction_forward(clip_feature)        # [B, D, Hc, Wc]
                elif "rn" in self.backbone.model_name.lower():
                    dense_proj = self.backbone.visual_prediction_forward(clip_feature)        # [B, D, Hc, Wc]
                else:
                    raise NotImplementedError

                # Normalize both sides for cosine-sim, then scale by CLIP logit_scale (like standard CLIP)
                dense_proj = F.normalize(dense_proj, dim=1)                                   # [B, D, Hc, Wc]
                text_cls   = F.normalize(text_classifier, dim=-1)                              # [C, D] (same order as mask_sem_results; last is void)
                logit_scale = self.backbone.clip_model.logit_scale.exp()

                # Per-pixel class logits at CLIP stride, then upsample to semantic resolution
                out_vocab_sem_results = logit_scale * torch.einsum("bdhw,kd->bkhw", dense_proj, text_cls)  # [B, C, Hc, Wc]
                out_vocab_sem_results = F.interpolate(
                    out_vocab_sem_results,
                    size=mask_sem_results.shape[-2:], mode="bilinear", align_corners=False
                )  # [B, C, H, W]

                # Max ensembling over templates
                final_pred_logits = []
                cur_idx = 0
                # Process each group of templates
                for num_t in num_templates:
                    # Slice current template group and take max
                    group_logits = out_vocab_sem_results[:, cur_idx:cur_idx + num_t]
                    final_pred_logits.append(group_logits.max(1).values)
                    cur_idx += num_t
                # Stack along new class dimension
                out_vocab_sem_results = torch.stack(final_pred_logits, dim=1)

                # 2) Geometric ensemble with in-vocab per-pixel logits (mask_sem_results)
                #    Follows your instance logic: seen uses alpha, unseen uses beta, and we carry through the in-vocab void prob.

                # Convert to probabilities
                in_probs_nv  = F.softmax(mask_sem_results,  dim=1).clamp_min(1e-8)
                out_probs_nv = F.softmax(out_vocab_sem_results, dim=1).clamp_min(1e-8)

                # Broadcastable seen/unseen mask over non-void classes
                ov = self.category_overlapping_mask.to(self.device)
                ov = ov.view(1, -1, 1, 1).to(in_probs_nv.dtype) if ov.dim() == 1 else ov.to(in_probs_nv.dtype)

                # Optional valid-pixel gating (pixel-wise analogue of your instance valid-mask gating)
                if self.ensemble_on_valid_mask:
                    # Trust out-vocab only where the pixel isn't predicted as void by in-vocab argmax
                    valid_masking = (mask_sem_results.argmax(dim=1, keepdim=True) != (mask_sem_results.shape[1] - 1)).to(in_probs_nv.dtype)
                    alpha_map = torch.ones_like(in_probs_nv) * self.geometric_ensemble_alpha * valid_masking
                    beta_map  = torch.ones_like(in_probs_nv) * self.geometric_ensemble_beta  * valid_masking
                else:
                    alpha_map = torch.ones_like(in_probs_nv) * self.geometric_ensemble_alpha
                    beta_map  = torch.ones_like(in_probs_nv) * self.geometric_ensemble_beta

                # Geometric mixing per pixel/class
                seen_mix   = (in_probs_nv.pow(1.0 - alpha_map) * out_probs_nv.pow(alpha_map)).clamp_min(1e-8)
                unseen_mix = (in_probs_nv.pow(1.0 - beta_map)  * out_probs_nv.pow(beta_map)).clamp_min(1e-8)

                # Blend seen vs unseen, renormalize non-void, then reattach void like your original logic
                cls_probs_nv = seen_mix * ov + unseen_mix * (1.0 - ov)                               # [B, C-1, H, W]
                cls_probs_nv = cls_probs_nv / cls_probs_nv.sum(dim=1, keepdim=True).clamp_min(1e-8)  # safe normalize
            
                mask_sem_results = torch.log(cls_probs_nv.clamp_min(1e-8))  # final ensembled semantic logits

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if mask_sem_results is not None:
                mask_sem_results = F.interpolate(
                    mask_sem_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

            del outputs

            # Prepare an iterator for boxes, which could be None
            if mask_box_results is None:
                mask_box_results = [None] * len(batched_inputs)
            if mask_round_results is None:
                mask_round_results = [None] * len(batched_inputs)
            if mask_sem_results is None:
                mask_sem_results = [None] * len(batched_inputs)
            if mask_cls_results is None:
                mask_cls_results = [None] * len(batched_inputs)

            processed_results = []
            for mask_cls_result, mask_obj_result, mask_pred_result, mask_sem_result, mask_box_result, mask_round_result, input_per_image, image_size in zip(
                mask_cls_results, mask_obj_results, mask_pred_results, mask_sem_results, mask_box_results, mask_round_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_sem_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_sem_result, image_size, height, width
                    ) if mask_sem_result is not None else None
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )

                # semantic segmentation inference
                if self.semantic_on:
                    if mask_sem_result is not None:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, mask_sem_result)
                    else:
                        r = mask_sem_result
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, mask_sem_result, mask_obj_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_sem_result, mask_obj_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            attributes = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
            if targets_per_image.has("gt_boxes"):
                h, w = targets_per_image.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                attributes["boxes"] = box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            new_targets.append(attributes)
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, mask_sem):
        if mask_sem is not None:
            return mask_sem.softmax(dim=0)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, mask_sem, mask_obj):
        # Obtain query class prediction based on semantic classification of
        # the mask area.
        if mask_cls is None:
            mask = mask_pred.softmax(dim=0)
            area = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8 # (B,Q)

            mask_cls = torch.einsum(
                "chw,qhw->qc", 
                mask_sem, 
                mask / area
            )

        # Obtain classification scores and objectness scores.
        cls_scores, labels = F.sigmoid(mask_cls).max(-1)
        objectness = F.sigmoid(mask_obj).squeeze(-1) # (B,Q)

        mask_pred = mask_pred.sigmoid()
        keep = objectness > 0.0
        cur_cls_scores = cls_scores[keep]
        cur_objectness = objectness[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_masks * cur_objectness.unsqueeze(-1).unsqueeze(-1) * cur_cls_scores.unsqueeze(-1).unsqueeze(-1)

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < 0.5:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_sem, mask_obj):
        device = mask_pred.device
        image_size = mask_pred.shape[-2:]  # (H, W)

        # Class prediction via semantic voting
        if mask_cls is None:
            mask = mask_pred.softmax(dim=0)
            area = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8 # (B,Q)

            mask_cls = torch.einsum(
                "chw,qhw->qc", 
                mask_sem, 
                mask / area
            )

        # --- 2) Objectness & labels like panoptic_inference ---
        labels = torch.sigmoid(mask_cls).argmax(-1)  # [Q]
        objectness = torch.sigmoid(mask_obj).view(-1)  # [Q]
        masks_sig = torch.sigmoid(mask_pred)  # [Q,H,W]

        # In panoptic_inference they compare against the number of *stuff* classes and drop "void"
        # Keep this behavior for consistency, then later filter to 'thing' for instances.
        keep = objectness > 0.0
        cur_objectness = objectness[keep]
        cur_classes = labels[keep]
        cur_masks = masks_sig[keep]  # [Qk,H,W]

        # Early exit if nothing survived
        result = Instances(image_size)
        if cur_masks.numel() == 0:
            # Empty Instances (all fields empty tensors)
            result.scores = torch.empty(0, device=device)
            result.pred_classes = torch.empty(0, dtype=torch.long, device=device)
            result.pred_masks = torch.empty(0, *image_size, dtype=torch.bool, device=device)
            result.pred_boxes = BitMasks(torch.empty(0, *image_size, dtype=torch.bool, device=device)).get_bounding_boxes()
            return result

        # --- 3) Argmax competition with objectness-weighted masks ---
        # Same as panoptic: weight masks by objectness, then per-pixel argmax across queries
        cur_prob_masks = cur_masks * cur_objectness.view(-1, 1, 1)  # [Qk,H,W]
        winner_ids = cur_prob_masks.argmax(0)  # [H,W], index in [0..Qk-1]

        # --- 4) Build per-instance outputs (thing-only) with area checks ---
        thing_ids = set(self.test_metadata.thing_dataset_id_to_contiguous_id.values())

        pred_masks_list = []
        pred_classes_list = []
        scores_list = []

        for k in range(cur_classes.shape[0]):
            pred_class = int(cur_classes[k].item())
            is_thing = pred_class in thing_ids
            if not is_thing:
                continue  # Instances should only return things

            # mask chosen by argmax, gated by the underlying query mask confidence (>= 0.5)
            winner_mask = (winner_ids == k)
            original_mask = (cur_masks[k] >= 0.5)

            mask_area = int(winner_mask.sum().item())
            original_area = int(original_mask.sum().item())
            if mask_area == 0 or original_area == 0:
                continue

            # Same consistency check as panoptic_inference
            if mask_area / max(original_area, 1) < 0.5:
                continue

            final_mask = winner_mask & original_mask
            if final_mask.sum().item() == 0:
                continue

            # Score: combine objectness with average mask prob over the final region
            avg_mask_prob = (cur_masks[k][final_mask].mean() if final_mask.any() else torch.tensor(0., device=device))
            score = cur_objectness[k] * avg_mask_prob

            pred_masks_list.append(final_mask)
            pred_classes_list.append(pred_class)
            scores_list.append(score)

        if len(pred_masks_list) == 0:
            # Return a valid empty Instances
            result.scores = torch.empty(0, device=device)
            result.pred_classes = torch.empty(0, dtype=torch.long, device=device)
            result.pred_masks = torch.empty(0, *image_size, dtype=torch.bool, device=device)
            result.pred_boxes = BitMasks(torch.empty(0, *image_size, dtype=torch.bool, device=device)).get_bounding_boxes()
            return result

        pred_masks = torch.stack(pred_masks_list, dim=0).to(torch.bool)  # [N,H,W]
        pred_classes = torch.tensor(pred_classes_list, dtype=torch.long, device=device)
        scores = torch.stack(scores_list).to(device)

        # --- 5) Pack into Instances ---
        result.pred_masks = pred_masks.float()  # keep float for compatibility with BitMasks/bboxes if needed
        result.pred_boxes = BitMasks(pred_masks).get_bounding_boxes()
        result.pred_classes = pred_classes
        result.scores = scores

        return result