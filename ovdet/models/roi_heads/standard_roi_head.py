# Copyright (c) OpenMMLab. All rights reserved.
import torch
import os
import numpy as np
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead
from mmengine.structures import InstanceData
from ovdet.methods.builder import OVD
from ovdet.models.attention.selfattention import TransformerEncoderLayer
from ovdet.models.attention.global_local_attention import TransformerEncoderLayer1
from ovdet.models.attention.global_local_window_attention import TransformerEncoderLayer2
from ovdet.models.attention.global_local_DC_attention import TransformerEncoderLayer3
from ovdet.models.attention.global_local_adaptive_attention import TransformerEncoderLayer4

@MODELS.register_module()
class OVDStandardRoIHead(StandardRoIHead):
    def __init__(self, transformer, clip_cfg=None, ovd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.transformer = TransformerEncoderLayer(**transformer)
        # self.transformer1 = TransformerEncoderLayer1(**transformer)
        # self.transformer2 = TransformerEncoderLayer2(**transformer)
        self.transformer3 = TransformerEncoderLayer3(**transformer)
        # self.transformer4 = TransformerEncoderLayer4(**transformer)
        if clip_cfg is None:
            self.clip = None
        else:
            self.clip = MODELS.build(clip_cfg)
        if ovd_cfg is not None:
            for k, v in ovd_cfg.items():
                # self.register_module(k, OVD.build(v))   # not supported in pt1.8.1
                setattr(self, k, OVD.build(v))

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, self.clip)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name, batch_inputs,
                *args, **kwargs):
        ovd_method = getattr(self, ovd_name)

        # -------------------------------------------------
        # 筛选之后的topk proposal 已经采样完了 ，确实是先采样 !!
        sampling_results_list = list(map(ovd_method.sample, rpn_results_list, batch_data_samples))
        if isinstance(sampling_results_list[0], InstanceData):
            rois = bbox2roi([res.bboxes for res in sampling_results_list])
        else:
            sampling_results_list_ = []
            bboxes = []
            for sampling_results in sampling_results_list:
                bboxes.append(torch.cat([res.bboxes for res in sampling_results]))
                sampling_results_list_ += sampling_results
            rois = bbox2roi(bboxes)
            sampling_results_list = sampling_results_list_
        # 经过roi之后变成7*7的feat
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.vision_to_language(bbox_feats)
        indi_embedding = region_embeddings.view(region_embeddings.size(0), -1)
        indi_embedding = self.bbox_head.fc_512_1(indi_embedding)
        return ovd_method.get_losses(region_embeddings,  sampling_results_list, self.clip, batch_inputs,
                                     indi_embedding, self.transformer3)
