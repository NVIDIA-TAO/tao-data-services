# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle kpi calculations."""
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool


def iou(boxes1, boxes2, border_pixels='half'):
    """Numpy version of element-wise iou.

    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Args:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates
            for one box in the format specified by `coords` or a 2D Numpy array of shape
            `(m, 4)` containing the coordinates for `m` boxes. If `mode` is set to 'element_wise',
            the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for
            one box in the format specified by `coords` or a 2D Numpy array of shape `(n, 4)`
            containing the coordinates for `n` boxes. If `mode` is set to 'element_wise', the
            shape must be broadcast-compatible with `boxes1`.`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float
        containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and
        `boxes2`. 0 means there is no overlap between two given boxes, 1 means their
        coordinates are identical.
    """
    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError(f"boxes1 must have rank either 1 or 2, but has rank {boxes1.ndim}.")
    if boxes2.ndim > 2:
        raise ValueError(f"boxes2 must have rank either 1 or 2, but has rank {boxes2.ndim}.")

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("Boxes list last dim should be 4 but got shape "
                         f"{boxes1.shape} and {boxes2.shape}, respectively.")

    # Set the correct coordinate indices for the respective formats.
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    # Compute the union areas.
    d = 0
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    # Compute the IoU.

    min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
    max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy + d)

    intersection_areas = side_lengths[:, 0] * side_lengths[:, 1]

    boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
    boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def _per_img_match(x, n_classes, sorting_algorithm, matching_iou_threshold, min_area):
    """
    Helper function for multithreading matching.

    Do not call this function from outside. It's outside the class definition purely due to python
    pickle issue.

    Arguments:
        x (tuple): (gt_box, pred_box)
        n_classes (int): number of classes
        sorting_algorithm (str): Which sorting algorithm the matching algorithm should
            use. This argument accepts any valid sorting algorithm for Numpy's `argsort()`
            function. You will usually want to choose between 'quicksort' (fastest and most
            memory efficient, but not stable) and 'mergesort' (slight slower and less memory
            efficient, but stable). The official Matlab evaluation algorithm uses a stable
            sorting algorithm, so this algorithm is only guaranteed to behave identically if you
            choose 'mergesort' as the sorting algorithm, but it will almost always behave
            identically even if you choose 'quicksort' (but no guarantees).
        matching_iou_threshold (float): A prediction will be considered a true
            positive if it has a Jaccard overlap of at least `matching_iou_threshold` with any
            ground truth bounding box of the same class.
    """
    gt = x[0]
    pred = x[1]
    if len(gt) == 0 and len(pred) == 0:
        TN = 1
    else:
        TN = 0

    # if gt is [], add a fake gt of class "-1"
    if len(gt) == 0:
        gt = np.array([[-1, 0, 0, 0, 0]])
    T = [[] for _ in range(n_classes)]
    P = [[] for _ in range(n_classes)]
    assert len(gt) > 0, "gt is empty"
    gt_cls = [gt[gt[:, 0] == i, 1:] for i in range(n_classes)]
    gt_cls_valid = [np.ones((len(i), )) for i in gt_cls]
    if len(pred) > 0:
        desc_inds = np.argsort(-pred[:, 1], kind=sorting_algorithm)
        pred = pred[desc_inds]
    else:
        # force 0. same as class_map
        pred_cls = 0

    for pred_box in pred:
        pred_cls = int(pred_box[0])
        P[pred_cls].append(pred_box[1])
        if len(gt_cls[pred_cls]) == 0:
            T[pred_cls].append(0)
            continue

        overlaps = iou(boxes1=gt_cls[pred_cls], boxes2=pred_box[2:])
        overlaps_unmatched = overlaps * gt_cls_valid[pred_cls]

        if np.max(overlaps_unmatched) >= matching_iou_threshold:
            gt_idx_in_cls = np.argmax(overlaps_unmatched)

            height = gt_cls[pred_cls][gt_idx_in_cls][3] - gt_cls[pred_cls][gt_idx_in_cls][1]
            width = gt_cls[pred_cls][gt_idx_in_cls][2] - gt_cls[pred_cls][gt_idx_in_cls][0]
            diff_criteria_matched = height * width < min_area
            if diff_criteria_matched:
                P[pred_cls].pop()
                continue

            T[pred_cls].append(1)
            # invalidate the matched gt
            gt_cls_valid[pred_cls][np.argmax(overlaps_unmatched)] = 0.0
        else:
            T[pred_cls].append(0)

    for idx, cls_valid in enumerate(gt_cls_valid):
        cls_valid = cls_valid.astype(int)
        non_match_count = 0
        for box_idx, box_valid in enumerate(cls_valid):
            if box_valid == 1:
                height = gt_cls[pred_cls][box_idx][3] - gt_cls[pred_cls][box_idx][1]
                width = gt_cls[pred_cls][box_idx][2] - gt_cls[pred_cls][box_idx][0]
                diff_criteria_matched = height * width < min_area
                if not diff_criteria_matched:
                    non_match_count += 1

        T[idx].extend([1] * non_match_count)
        P[idx].extend([0.0] * non_match_count)

    return (T, P, TN)


def voc_ap(
        rec,
        prec,
        average_precision_mode,
        num_recall_points):
    """Calculate VOC style AP."""
    if average_precision_mode == 'sample':
        ap = 0.
        for t in np.linspace(0., 1.0, num_recall_points):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / float(num_recall_points)
    elif average_precision_mode == 'integrate':
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    else:
        raise ValueError("average_precision_mode should be either sample or integrate")
    return ap


def evaluate_class(per_img_match, gt_list, pred_list, conf_threshold):
    """Evaluate per class."""
    TP = 0.
    FP = 0.
    FN = 0.
    TN = 0.  # Warning: TN is calculated per image.
    prec, rec = [], []

    # Start multiprocessing
    with Pool() as pool:
        results = pool.map(per_img_match, list(zip(gt_list, pred_list)))

    TT = [[] for _ in range(1)]
    PP = [[] for _ in range(1)]

    for t, p, tn in results:
        for i in range(1):
            TT[i] += t[i]
            PP[i] += p[i]
            TN += tn

    for T, P in zip(TT, PP):
        # sort according to prob.
        Ta = np.array(T)
        Pa = np.array(P)
        s_idx = np.argsort(-Pa, kind='quicksort')
        P = Pa[s_idx].tolist()
        T = Ta[s_idx].tolist()
        npos = np.sum(Ta)
        for t, p in zip(T, P):
            if t == 1 and p >= conf_threshold:
                TP += 1
            elif t == 1 and p < conf_threshold:
                FN += 1
            elif t == 0 and p >= conf_threshold:
                FP += 1
            if TP + FP == 0.:
                precision = 0.
            else:
                precision = float(TP) / (TP + FP)
            if npos > 0:
                recall = float(TP) / float(npos)
            else:
                recall = 0.0
            prec.append(precision)
            rec.append(recall)

    # To prevent divison by zero
    delta = 1e-7
    kpi_precision = TP / (TP + FP + delta) if TP + FP > 0 else 0
    kpi_recall = TP / (TP + FN + delta) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN + delta)

    return TP, FP, FN, TN, kpi_precision, kpi_recall, accuracy, prec, rec


def evaluate(gt_data_obj, pred_data_obj, cat_map, matching_iou_threshold=0.5, conf_threshold=0.5, ignore_sqwidth=40, num_recall_points=11):
    """ Evaluate KITTI style KPI.

    Args:
        gt_data_obj (DataFormat): Object for ground truth data.
        pred_data_obj (DataFormat): Object for predicted data.
        cat_map (OrderedDict): Category Mapping for KITTI data
        matching_iou_threshold (float): IOU threshold.
        conf_threshold (float): confidence threshold.
        ignore_sqwidth (int): ignore GT & Pred that have area smaller than this value
        num_recall_points (int): eleven is default

    Return:
        df(Pandas Data Frame), MAP(float): KPI values and MAP value.
    """
    gt_df = gt_data_obj.df
    pred_df = pred_data_obj.df

    if cat_map:
        class_map = {}
        for k, classes in cat_map.items():
            for c in classes:
                class_map[c] = k
        class_list = list(cat_map.keys())

        # Map GT based on the category mapping
        gt_df.type = gt_df.type.map(lambda x: class_map[x] if x in class_map else None)
        gt_df = gt_df[gt_df.type.isin(class_list)]

    min_area = ignore_sqwidth ** 2

    stats = []

    # Iterate over each object class data for KPI calculation.
    for class_name in class_list:
        cls_gt_df = gt_df[gt_df.type == class_name]
        # Enforce 0
        cls_gt_df['type'].values[:] = 0

        cls_pred_df = pred_df[pred_df.type == class_name]
        # Enforce 0
        cls_pred_df['type'].values[:] = 0

        # Prepare for multiprocessing
        per_img_match = partial(_per_img_match, n_classes=1,
                                sorting_algorithm='quicksort',
                                matching_iou_threshold=matching_iou_threshold,
                                min_area=min_area)

        # Loop through every img
        gt_list, pred_list = [], []
        for img_id in gt_data_obj.ids:
            per_img_gt_df = cls_gt_df[cls_gt_df.img_name == img_id]
            per_img_pred_df = cls_pred_df[cls_pred_df.img_name == img_id]
            per_img_gt_arr = per_img_gt_df[['type', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].to_numpy().astype(float)
            per_img_pred_arr = per_img_pred_df[['type', 'conf_score', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].to_numpy().astype(float)

            # Filter predictions based on minimum area
            mask = (per_img_pred_arr[:, 5] - per_img_pred_arr[:, 3]) * (per_img_pred_arr[:, 4] - per_img_pred_arr[:, 2]) >= min_area
            per_img_pred_arr = per_img_pred_arr[mask]

            # Filter predictions based on confidence threshold
            per_img_pred_arr = per_img_pred_arr[per_img_pred_arr[:, 1] >= conf_threshold]
            gt_list.append(per_img_gt_arr)
            pred_list.append(per_img_pred_arr)

        # Generate statistics for the current class
        TP, FP, FN, TN, precision, recall, accuracy, prec, rec = evaluate_class(per_img_match, gt_list, pred_list, conf_threshold)

        # Calculate Pascal VOC style AP
        ap = voc_ap(np.array(prec), np.array(rec), "sample", num_recall_points)

        stats.append({
            'class_name': class_name,
            'Pr': precision,
            'Re': recall,
            'precision': prec,  # Used for PR Curves
            'recall': rec,  # Used for PR Curves
            'AP': ap,
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
            'TN': int(TN),
            'Acc': accuracy
        })

    # Calculate mAP
    mAP = np.mean([v['AP'] for v in stats])

    df = pd.DataFrame(stats)
    return df, mAP
