#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from mum import add_ubteacher_config
from mum.engine.trainer import UBTeacherTrainer, BaselineTrainer, MUMTrainer

# hacky way to register
from mum.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from mum.modeling.proposal_generator.rpn import PseudoLabRPN
from mum.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from mum.modeling.swin import MySwin
from mum.modeling.fpn_tut import build_resnet_fpn_backbone_tut
import mum.data.datasets.builtin

from mum.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from detectron2.data.datasets import register_coco_instances

register_coco_instances("TRAIN_DATASET", {}, "data_full_size/coco/annotations/instances_train2017.json", "data_full_size/coco/train")
register_coco_instances("VAL_DATASET", {}, "data_full_size/coco/annotations/instances_val2017.json", "data_full_size/coco/valid")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    elif cfg.SEMISUPNET.Trainer == "mum":
        Trainer = MUMTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher" or cfg.SEMISUPNET.Trainer=='mum':
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
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
