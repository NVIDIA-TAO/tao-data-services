# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO inference."""

from copy import deepcopy
import os
import json
import shutil
import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import torch
from torchvision.ops import batched_nms

from lightning_fabric import Fabric
from lightning_fabric.strategies import DDPStrategy

from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.deformable_detr.utils.box_ops import box_cxcywh_to_xyxy

from nvidia_tao_pytorch.cv.grounding_dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.grounding_dino.model.post_process import PostProcess
from nvidia_tao_pytorch.cv.grounding_dino.model.utils import grounding_dino_parser, ptm_adapter

from nvidia_tao_ds.auto_label.grounding_dino.utils import (
    plot_boxes_to_image, get_json_result, save_results, load_jsonlines
)
from nvidia_tao_ds.auto_label.grounding_dino.tokenize import create_positive_map, tokenize_captions
from nvidia_tao_ds.auto_label.grounding_dino.dataset import setup_dataloader
from nvidia_tao_ds.annotations.merger import ODVGMerger


def load_model(experiment_config, checkpoint_path):
    """Load Grounding DINO model"""
    model = build_model(experiment_config=experiment_config)
    checkpoint = load_pretrained_weights(checkpoint_path, ptm_adapter=ptm_adapter, parser=grounding_dino_parser)
    new_checkpoint = {k.replace("model.model.", "model."): v for k, v in checkpoint.items()}
    model.load_state_dict(new_checkpoint)
    return model


def iterate_single_grounding(model, dataloader, results_dir,
                             box_threshold=0.5, nms_threshold=0.7,
                             visualize=False, ann_type="detection"):
    """Grounding / phrase auto-labeling."""
    results, remaining_results = [], []
    special_tokens = model.model.specical_tokens
    tokenizer = model.model.tokenizer
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            data, targets, image_names = batch

            captions = [t["caption"] for t in targets]

            (
                tokenized,
                _,
                position_ids,
                text_self_attention_masks
            ) = tokenize_captions(captions, special_tokens, tokenizer)

            outputs = model(data,
                            input_ids=tokenized["input_ids"],
                            attention_mask=tokenized["attention_mask"],
                            position_ids=position_ids,
                            token_type_ids=tokenized["token_type_ids"],
                            text_self_attention_masks=text_self_attention_masks)

            out_logits, out_bboxes = outputs['pred_logits'], outputs['pred_boxes']
            out_logits = out_logits.sigmoid()

            # Post-Process
            batch_results = []
            for caption, target, out_logit, out_bbox, image_name in zip(captions,
                                                                        targets,
                                                                        out_logits,
                                                                        out_bboxes,
                                                                        image_names):
                cat_list = target["cat_list"]
                target_size = target["orig_size"]
                full_caption = target["full_caption"]

                tok = model.model.tokenizer(caption, padding="longest", return_tensors="pt")
                label_list = torch.arange(len(cat_list))
                pos_maps = create_positive_map(tok, label_list, cat_list, caption)

                prob_to_token = out_logit
                pos_maps = pos_maps.to(prob_to_token.device)
                for label_ind in range(len(pos_maps)):
                    if pos_maps[label_ind].sum() != 0:
                        pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

                logits_for_phrases = (prob_to_token @ pos_maps.T).T

                # Filter out predictions
                all_logits, all_labels, all_boxes, all_indices = [], [], [], []
                remaining_phrases = deepcopy(cat_list)
                for phrase, logit_phr in zip(cat_list, logits_for_phrases):
                    filt_mask = logit_phr > box_threshold
                    # Filter box
                    all_boxes.append(out_bbox[filt_mask])
                    # Filter logits
                    logit_phr_num = logit_phr[filt_mask]
                    all_logits.append(logit_phr_num)
                    all_labels.extend([phrase + f"({logit.item():.2f})" for logit in logit_phr_num])
                    all_indices.extend([cat_list.index(phrase) for _ in logit_phr_num])

                    if len(logit_phr_num):
                        # Remove the detected phrase from the list
                        remaining_phrases.remove(phrase)

                if len(all_boxes):
                    # Scale prediction back to xyxy at the original resolution
                    boxes_filt = torch.cat(all_boxes, dim=0)
                    scale_fct = torch.tensor([target_size[1], target_size[0],
                                              target_size[1], target_size[0]] * len(boxes_filt))
                    scale_fct = scale_fct.reshape((len(boxes_filt), 4)).to(boxes_filt.device)
                    boxes_filt = boxes_filt * scale_fct
                    boxes_filt = box_cxcywh_to_xyxy(boxes_filt)

                    # Run NMS
                    if nms_threshold:
                        scores = torch.cat(all_logits, dim=0)
                        indices = torch.tensor(all_indices)
                        keep_ind = batched_nms(boxes_filt, scores, indices, nms_threshold)

                        boxes_filt = boxes_filt[keep_ind]

                        all_labels = np.array(all_labels)[keep_ind.cpu().numpy()]
                        all_logits = scores[keep_ind]

                    pred_dict = {
                        "image_name": image_name,
                        "caption": full_caption,
                        "size": target_size.cpu().tolist(),
                        "boxes": boxes_filt.cpu().tolist(),
                        "labels": all_labels,
                        "scores": all_logits if isinstance(all_logits, list) else all_logits.cpu()
                    }
                    batch_results.append(pred_dict)

                # Now also keep record of remaining phrases
                remaining_results.append({"file_name": os.path.basename(image_name),
                                          "caption": full_caption,
                                          "noun_chunks": remaining_phrases})

            for result in batch_results:
                if visualize:
                    pil_input = Image.open(result['image_name']).convert("RGB")
                    pil_input = ImageOps.exif_transpose(pil_input)

                    output_image_name = os.path.join(results_dir, "images_annotated",
                                                     os.path.basename(result['image_name']))
                    image_with_box = plot_boxes_to_image(pil_input, result)
                    image_with_box.save(output_image_name)
                results.append(get_json_result(result, ann_type=ann_type))
    return results, remaining_results


def iterate_single(model, dataloader, results_dir, captions, categories, box_processors,
                   box_threshold=0.5, nms_threshold=0.7, visualize=False):
    """Closed-set category auto-labeling."""
    results, remaining_results = [], []
    special_tokens = model.model.specical_tokens
    tokenizer = model.model.tokenizer
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            data, targets, image_names = batch
            batch_caption = [' . '.join(captions) + ' .'] * len(data)
            (
                tokenized,
                _,
                position_ids,
                text_self_attention_masks
            ) = tokenize_captions(batch_caption, special_tokens, tokenizer)

            outputs = model(data,
                            input_ids=tokenized["input_ids"],
                            attention_mask=tokenized["attention_mask"],
                            position_ids=position_ids,
                            token_type_ids=tokenized["token_type_ids"],
                            text_self_attention_masks=text_self_attention_masks)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            predictions = box_processors(outputs, orig_target_sizes, image_names)

            for image_name, target_size, prediction in zip(image_names, orig_target_sizes, predictions):
                filt_mask = prediction['scores'] > box_threshold
                scores = prediction["scores"][filt_mask]
                label_ids = prediction["labels"][filt_mask]
                boxes = prediction["boxes"][filt_mask]

                if len(boxes):
                    # Run NMS
                    if nms_threshold:
                        keep_ind = batched_nms(boxes, scores, label_ids, nms_threshold)

                        boxes = boxes[keep_ind]
                        label_ids = label_ids[keep_ind]
                        scores = scores[keep_ind]

                    labels = [categories[int(lab.item())] for lab in label_ids]
                    target_size = target_size.cpu().tolist()
                    pred_dict = {
                        "image_name": image_name,
                        "size": [target_size[0], target_size[1]],
                        "boxes": boxes.cpu().tolist(),
                        "labels": labels,
                        "label_ids": label_ids.cpu().tolist(),
                        "scores": scores.cpu().tolist()
                    }
                    results.append(get_json_result(pred_dict, ann_type="detection"))

                    # Now also keep record of remaining phrases
                    remaining_phrase = list(set(categories) - set(labels))
                    remaining_results.append({"file_name": os.path.basename(image_name),
                                              "caption": " . ".join(categories) + " .",
                                              "noun_chunks": remaining_phrase})

                    if visualize:
                        pil_input = Image.open(image_name).convert("RGB")
                        pil_input = ImageOps.exif_transpose(pil_input)

                        pred_dict["labels"] = \
                            [f"{lb}({sc:.2f})" for lb, sc in zip(pred_dict["labels"], pred_dict["scores"])]

                        output_image_name = os.path.join(results_dir,
                                                         "images_annotated",
                                                         os.path.basename(image_name))
                        image_with_box = plot_boxes_to_image(pil_input, pred_dict)
                        image_with_box.save(output_image_name)

    return results, remaining_results


def run_grounding_inference(experiment_config, results_dir):
    """Automatically generate bounding boxes using Grounding DINO."""
    # Align CUDA and TAO visible devices and prefer NCCL for GPU DDP
    tao_visible = os.environ.get('TAO_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    gpu_ids = [int(gpu) for gpu in tao_visible.split(',') if gpu != '']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)

    using_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
    strategy = DDPStrategy(process_group_backend='nccl') if using_gpu else 'ddp'
    accelerator = 'gpu' if using_gpu else 'cpu'
    num_devices = max(1, len(gpu_ids)) if using_gpu else 1

    fabric = Fabric(accelerator=accelerator, devices=num_devices, strategy=strategy)
    fabric.launch()

    batch_size = experiment_config.batch_size
    num_workers = experiment_config.num_workers

    # Now we override everything to grounding dino
    experiment_config = experiment_config.grounding_dino
    checkpoint_path = experiment_config.checkpoint
    original_results_dir = results_dir

    visualize = experiment_config.visualize
    dataset_config = experiment_config.dataset
    root_dir = dataset_config.image_dir

    is_grounding, is_closed = False, False
    json_file, class_names = None, None
    if dataset_config.get("noun_chunk_path", None):
        is_grounding = True
        json_file = dataset_config.noun_chunk_path
        fabric.print("Processing grounding annotation")

    if dataset_config.get("class_names", None):
        is_closed = True
        class_names = list(dataset_config.class_names)
        fabric.print("Processing closed category annotation")

    # Check config to ensure that only either config is passed.
    if is_grounding and is_closed:
        raise ValueError("Can't set both dataset.json_file and "
                         "dataset.class_names at once. Only provide either of them. "
                         f"json_file: {json_file} class_names: {class_names}")

    # To disable warnings from HF tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model
    model = load_model(experiment_config, checkpoint_path)
    model = fabric.setup(model)
    model.eval()

    resultdir_lists = []
    for idx, schedule in enumerate(experiment_config.iteration_scheduler):
        if idx != 0:
            json_file = os.path.join(results_dir, "remaining.jsonl")
            fabric.print(f"Input file: {json_file}")
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"{json_file}")

        # Update results dir
        results_dir = os.path.join(original_results_dir, f"auto_label{idx}")
        os.makedirs(results_dir, exist_ok=True)

        if visualize:
            os.makedirs(os.path.join(results_dir, "images_annotated"), exist_ok=True)

        box_threshold = schedule["conf_threshold"]
        nms_threshold = schedule["nms_threshold"]

        if is_closed and idx == 0:
            # First iteration of closed-set category auto-labeling
            # can be conisdered as a regular inference
            box_processors = PostProcess(model.model.tokenizer,
                                         cat_list=class_names,
                                         num_select=100)
            dataloader = setup_dataloader(root_dir=root_dir,
                                          captions=class_names,
                                          augmentation=dataset_config.augmentation,
                                          batch_size=batch_size,
                                          num_workers=num_workers)
            dataloader = fabric.setup_dataloaders(dataloader)

            results, remaining_results = iterate_single(model, dataloader, results_dir,
                                                        class_names, class_names, box_processors,
                                                        box_threshold, nms_threshold, visualize)
        else:
            dataloader = setup_dataloader(root_dir=root_dir,
                                          json_file=json_file,
                                          augmentation=dataset_config.augmentation,
                                          batch_size=batch_size,
                                          num_workers=num_workers)
            dataloader = fabric.setup_dataloaders(dataloader)

            if is_closed:
                results, remaining_results = iterate_single_grounding(model, dataloader, results_dir,
                                                                      box_threshold, nms_threshold,
                                                                      visualize, ann_type="detection-2")
            else:
                results, remaining_results = iterate_single_grounding(model, dataloader, results_dir,
                                                                      box_threshold, nms_threshold,
                                                                      visualize, ann_type="grounding")

        save_results(results, remaining_results, fabric, results_dir, multigpu=(len(gpu_ids) > 1))
        resultdir_lists.append(results_dir)
        fabric.print(f"Iteration #{idx + 1} completed")

        # Empty cuda for the next iteration
        torch.cuda.empty_cache()

    # Aggregate the file on global rank 0 only, then synchronize
    if fabric.global_rank == 0:
        final_annotation_path = os.path.join(original_results_dir, "final_annotation.jsonl")
        if len(experiment_config.iteration_scheduler) > 1:
            fabric.print("Merging each iteration into one")
            merger = ODVGMerger([os.path.join(r, "autolabelled.jsonl") for r in resultdir_lists])
            merger.merge(final_annotation_path)
        else:
            shutil.copy(os.path.join(resultdir_lists[0], "autolabelled.jsonl"), final_annotation_path)
        fabric.print(f"Annotation stored at {final_annotation_path}")

        if is_closed:
            category_map = {str(i): cat for i, cat in enumerate(class_names)}
            cmap_path = os.path.join(original_results_dir, "labelmap.json")
            with open(cmap_path, mode="w", encoding="utf-8") as f:
                json.dump(category_map, f)
            fabric.print(f"Category mapping stored at {cmap_path}")

        # Remove all jsonl files that have been temporarily stored for DDP runs
        for r in resultdir_lists:
            for file in glob.glob(os.path.join(r, "*.rank*jsonl")):
                if os.path.exists(file):
                    os.remove(file)

        if visualize:
            os.makedirs(os.path.join(original_results_dir, "images_annotated"), exist_ok=True)
            fabric.print("Running visualization of the final merged annotation")

            final_results = load_jsonlines(final_annotation_path)

            for final_result in tqdm(final_results, total=len(final_results)):
                image_name = os.path.join(root_dir, final_result["file_name"])
                boxes, labels = [], []
                if is_closed:
                    for inst in final_result["detection"]["instances"]:
                        boxes.append(inst["bbox"])
                        labels.append(inst["category"])
                else:
                    for inst in final_result["grounding"]["regions"]:
                        boxes.append(inst["bbox"])
                        labels.append(inst["phrase"])

                pil_input = Image.open(image_name).convert("RGB")
                pil_input = ImageOps.exif_transpose(pil_input)

                output_image_name = os.path.join(original_results_dir,
                                                 "images_annotated",
                                                 os.path.basename(image_name))
                image_with_box = plot_boxes_to_image(pil_input, {"boxes": boxes, "labels": labels})
                image_with_box.save(output_image_name)

    fabric.barrier()
