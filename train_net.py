"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py

FCCLIP Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    hooks,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import build_batch_data_loader, trivial_batch_collator
import weakref

from fcclip import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    COCOPanopticEvaluator,
    COCOZSPanopticEvaluator,
    ExportEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    ZSSemSegEvaluator,
    add_maskformer2_config,
    add_fcclip_config,
    add_zegfc_config,
    build_compiled_model
)


def build_limited_detection_test_loader(cfg, dataset_name, mapper, max_eval_images):
    """Create an evaluation data loader that only iterates over a subset."""

    logger = logging.getLogger(__name__)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    total_images = len(dataset_dicts)

    if total_images == 0:
        logger.warning(
            "Dataset '%s' is empty. Returning the standard test loader.",
            dataset_name,
        )
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    if max_eval_images < total_images and comm.is_main_process():
        logger.info(
            "Limiting evaluation on '%s' to %d images (out of %d).",
            dataset_name,
            max_eval_images,
            total_images,
        )

    limited_dicts = list(dataset_dicts[:max_eval_images])

    if len(limited_dicts) < max_eval_images and comm.is_main_process():
        logger.warning(
            "Requested %d evaluation images, but dataset '%s' only has %d. Using all available images.",
            max_eval_images,
            dataset_name,
            len(limited_dicts),
        )

    dataset = DatasetFromList(limited_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=False)

    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(limited_dicts))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
    )


class MemEfficientDetectionCheckpointer(DetectionCheckpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.
        Only saves parameters that have gradients (non-frozen parameters).
        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return
        
        data = {}
        
        # Only save model parameters that require gradients
        model_state_dict = {}
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                model_state_dict[param_name] = param.data
        
        # Also include buffers (batch norm stats, etc.) as they're usually needed
        for buffer_name, buffer in self.model.named_buffers():
            model_state_dict[buffer_name] = buffer.data
            
        data["model"] = model_state_dict
        
        # For checkpointables, filter based on requires_grad if they have parameters
        for key, obj in self.checkpointables.items():
            if hasattr(obj, 'state_dict'):
                obj_state_dict = {}
                
                # If the object has named_parameters method (like optimizers, schedulers)
                if hasattr(obj, 'named_parameters'):
                    for param_name, param in obj.named_parameters():
                        if param.requires_grad:
                            obj_state_dict[param_name] = param.data
                else:
                    # For other checkpointables (optimizers, schedulers), save full state
                    # as they typically only store states for parameters that require gradients
                    obj_state_dict = obj.state_dict()
                    
                data[key] = obj_state_dict
            else:
                # If no state_dict method, save the object as-is
                data[key] = obj
        
        data.update(kwargs)
        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        
        # Log how many parameters are being saved vs total
        total_params = sum(p.numel() for p in self.model.parameters())
        saved_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Saving {saved_params}/{total_params} parameters ({saved_params/total_params*100:.1f}%)")
        
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)


class SwapProbabilitySchedulerHook(hooks.HookBase):
    """Linearly swap thing/stuff probabilities over the course of training."""

    def __init__(
        self,
        base_model,
        training_model,
        max_iter,
        start_thing: float,
        end_thing: float,
        start_stuff: float,
        end_stuff: float,
    ):
        self.base_model = base_model
        self.training_model = training_model
        self.max_iter = max_iter
        self.start_thing = start_thing
        self.end_thing = end_thing
        self.start_stuff = start_stuff
        self.end_stuff = end_stuff

    def _set_probabilities(self, progress: float):
        prob_swap_thing = max(
            0.0, min(1.0, self.start_thing + (self.end_thing - self.start_thing) * progress)
        )
        prob_swap_stuff = max(
            0.0, min(1.0, self.start_stuff + (self.end_stuff - self.start_stuff) * progress)
        )

        for model in (self.base_model, self.training_model):
            if model is None:
                continue

            target_model = model.module if hasattr(model, "module") else model

            if hasattr(target_model, "probability_swap_thing"):
                target_model.probability_swap_thing = prob_swap_thing
            if hasattr(target_model, "probability_swap_stuff"):
                target_model.probability_swap_stuff = prob_swap_stuff

    def _progress(self, iteration: int) -> float:
        # Ensure we handle edge cases like a single-iteration training run.
        denom = max(self.max_iter - 1, 1)
        return max(0.0, min(1.0, iteration / denom))

    def before_train(self):
        progress = self._progress(getattr(self.trainer, "start_iter", 0))
        self._set_probabilities(progress)

    def before_step(self):
        progress = self._progress(self.trainer.iter)
        self._set_probabilities(progress)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to FCCLIP.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model, compiled_model = self.build_model(cfg)
        self.base_model = model
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        compiled_model = create_ddp_model(compiled_model, broadcast_buffers=False)

        if cfg.SOLVER.AMP.ENABLED:
            # Convert precision string to torch dtype
            precision_map = {
                "float16": torch.float16,
                "float8_e4m3fn": torch.float8_e4m3fn,
                "float8_e5m2": torch.float8_e5m2
            }
            precision = precision_map.get(cfg.SOLVER.AMP.PRECISION, torch.float16)

            self._trainer = AMPTrainer(
                compiled_model, data_loader, optimizer,
                precision=precision
            )
        else:
            self._trainer = SimpleTrainer(
                compiled_model, data_loader, optimizer
            )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = MemEfficientDetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            SwapProbabilitySchedulerHook(
                self.base_model,
                self._trainer.model,
                self.max_iter,
                cfg.MODEL.ZEG_FC.PROBABILITY_SWAP_THING,
                cfg.MODEL.ZEG_FC.PROBABILITY_SWAP_THING_END,
                cfg.MODEL.ZEG_FC.PROBABILITY_SWAP_STUFF,
                cfg.MODEL.ZEG_FC.PROBABILITY_SWAP_STUFF_END,
            ),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            # TODO perform inference with the uncompiled model because PyTorch compile is
            # absolute fucking dogshit and cries and begs for mercy at the slightest inconvenience 
            self._last_eval_results = self.test(self.cfg, self.base_model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model, compiled_model = build_compiled_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model, compiled_model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "zs_sem_seg", "ade20k_panoptic_seg", "zs_ade20k_panoptic_seg"]:
            sem_seg_evaluator = (
                ZSSemSegEvaluator
                if evaluator_type in ["zs_sem_seg", "zs_ade20k_panoptic_seg"]
                else SemSegEvaluator
            )
            evaluator_list.append(
                sem_seg_evaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        elif evaluator_type in ["zs_sem_seg", "zs_ade20k_panoptic_seg"]:
            evaluator_list.append(
                ZSSemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type in [
            "zs_coco_panoptic_seg",
            "zs_ade20k_panoptic_seg",
            "zs_cityscapes_panoptic_seg",
            "zs_mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOZSPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in ["coco_panoptic_seg", "zs_coco_panoptic_seg"] and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            sem_seg_evaluator = (
                ZSSemSegEvaluator
                if evaluator_type == "zs_coco_panoptic_seg"
                else SemSegEvaluator
            )
            evaluator_list.append(
                sem_seg_evaluator(dataset_name, distributed=True, output_dir=output_folder)
            )
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "mapillary_vistas_panoptic_seg",
            "zs_mapillary_vistas_panoptic_seg",
        ] and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            sem_seg_evaluator = (
                ZSSemSegEvaluator
                if evaluator_type == "zs_mapillary_vistas_panoptic_seg"
                else SemSegEvaluator
            )
            evaluator_list.append(
                sem_seg_evaluator(dataset_name, distributed=True, output_dir=output_folder)
            )
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type in ["ade20k_panoptic_seg", "zs_ade20k_panoptic_seg"] and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
        
        # Output save
        if cfg.MODEL.MASK_FORMER.TEST.EXPORT_OUTPUTS:
            evaluator_list.append(ExportEvaluator(dataset_name, output_dir=output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = None
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, False)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, False)

        max_eval_images = cfg.MODEL.MASK_FORMER.TEST.MAX_EVAL_IMAGES
        if max_eval_images and max_eval_images > 0:
            return build_limited_detection_test_loader(cfg, dataset_name, mapper, max_eval_images)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    add_zegfc_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "fcclip" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="fcclip")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model, compiled_model = Trainer.build_model(cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        frozen_params_exclude_text = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                continue
            # ignore text tower
            if 'clip_model.token_embedding' in n or 'clip_model.positional_embedding' in n or 'clip_model.transformer' in n or 'clip_model.ln_final' in n or 'clip_model.text_projection' in n:
                continue
            frozen_params_exclude_text += p.numel()    
        print(f"total_params: {total_params}, trainable_params: {trainable_params}, frozen_params: {frozen_params}, frozen_params_exclude_text: {frozen_params_exclude_text}")

        MemEfficientDetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, compiled_model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, compiled_model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
