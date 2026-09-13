# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert AICity annotations to OVPKL format"""

import os
import json
import h5py
import copy
import pickle
import shutil
import subprocess
import tqdm
import numpy as np

from nvidia_tao_ds.core.logging.logging import logger
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from spatialai_data_utils.constants import FPS
from spatialai_data_utils.loaders.calibration import load_calib
try:
    # spatialai-data-utils < 2.0
    from spatialai_data_utils.utils.camera_name_utils import get_cam_names_in_scene
except ModuleNotFoundError:
    # spatialai-data-utils >= 2.0
    from spatialai_data_utils.datasets.scenes import get_cam_names_in_scene
try:
    # Removed in spatialai-data-utils 2.0; retained for legacy container images.
    from spatialai_data_utils.visualization.video_utils.video2frame import (
        video2frame_multi_cameras_syn,
    )
except ImportError:
    video2frame_multi_cameras_syn = None

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in qulity
YAW = 6  # decoded

AICITY_OVPKL_SCHEMA_VERSION = "aicity_ovpkl/v1"
_GENERATED_ANNOTATION_CACHE_FILENAMES = {
    "_pkl_cam_counts.pkl",
}
_H5_SUFFIXES = (".h5", ".hdf5")


def _is_h5_path(path):
    """Return whether a path uses a supported HDF5 filename suffix."""
    return str(path).lower().endswith(_H5_SUFFIXES)


def _camera_name_from_storage(storage_name):
    """Return the calibration camera name for a directory or HDF5 file."""
    storage_name = str(storage_name)
    if _is_h5_path(storage_name):
        return os.path.splitext(storage_name)[0]
    return storage_name


def _find_h5_filename(directory, camera_name):
    """Resolve an HDF5 camera filename without appending a duplicate suffix."""
    base_name = _camera_name_from_storage(os.path.basename(str(camera_name)))
    candidates = [f"{base_name}.h5", f"{base_name}.hdf5"]
    for candidate in candidates:
        if os.path.isfile(os.path.join(directory, candidate)):
            return candidate
    return candidates[0]


def _get_h5_depth_path(scene_path, scene_name, camera_storage_name, frame_id):
    """Return absolute/relative HDF5 depth references for one camera frame."""
    depth_filename_in_h5 = f"distance_to_image_plane_{frame_id:05}.png"
    depth_dir = os.path.join(scene_path, "depth_maps")
    camera_name = _camera_name_from_storage(camera_storage_name)
    if os.path.isdir(depth_dir):
        depth_filename = _find_h5_filename(depth_dir, camera_name)
        absolute_path = os.path.join(depth_dir, depth_filename)
        relative_path = os.path.join(scene_name, "depth_maps", depth_filename)
        depth_key = depth_filename_in_h5
    else:
        if _is_h5_path(camera_storage_name):
            depth_filename = str(camera_storage_name)
        else:
            depth_filename = _find_h5_filename(scene_path, camera_name)
        absolute_path = os.path.join(scene_path, depth_filename)
        relative_path = os.path.join(scene_name, depth_filename)
        depth_key = os.path.join(
            "distance_to_image_plane_png",
            depth_filename_in_h5,
        )
    return (absolute_path, depth_key), (relative_path, depth_key)


def _is_annotation_pkl(filename):
    """Return whether a filename is a scene annotation rather than an index."""
    return (
        filename.endswith(".pkl") and
        filename not in _GENERATED_ANNOTATION_CACHE_FILENAMES and
        not filename.endswith("_lazy_index.pkl")
    )


def convert_aicity_to_ovpkl(
    cfg=None,
    verbose=False
):
    """Convert AICity annotations to OVPKL format.

    Args:
        cfg (dict): Configuration.
        verbose (bool): Verbosity.
    """
    if cfg is not None:
        root_path = cfg.aicity.root
        version = cfg.aicity.version
        splits = cfg.aicity.split
        class_config = cfg.aicity.class_config
        recentering = cfg.aicity.recentering
        rgb_format = cfg.aicity.rgb_format
        depth_format = cfg.aicity.depth_format
        camera_grouping_modes = cfg.aicity.camera_grouping_mode
        anchor_init_config = cfg.aicity.anchor_init_config
        class_config = update_class_config(class_config)
        output_dir = cfg.results_dir
        num_frames = cfg.aicity.num_frames
    else:
        raise ValueError("config is not provided")

    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Root path {root_path} does not exist")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = splits.split(",")
    if rgb_format == "mp4":
        # convert mp4 to jpg
        video_to_frame(root_path, splits, num_frames=num_frames)
        rgb_format = "jpg"

    camera_group_config = None
    if len(camera_grouping_modes) > 0:
        camera_grouping_modes = camera_grouping_modes.split(",")
        if len(camera_grouping_modes) > 0:
            camera_group_config = {
                "use_camera_groups": True,
                "use_training_camera_groups": "training" in camera_grouping_modes,
                "use_random_camera_groups": "random" in camera_grouping_modes,
                "n_groups": 3,
                "n_cams_range_per_group": [5, 10],
            }
            logger.info(f"camera_group_config: {camera_group_config}")

    logger.info(f"camera_grouping_modes: {camera_grouping_modes}")
    logger.info(f"camera_group_config: {camera_group_config}")

    if version == "2025":
        for split in splits:
            create_ov_infos_aicity2025(
                root_path,
                output_dir,
                rgb_format=rgb_format,
                depth_format=depth_format,
                version=version,
                split=split,
                class_config=class_config,
                camera_group_config=camera_group_config,
                recentering=recentering,
                num_frames=num_frames,
                verbose=verbose,
            )
    elif version == "2024":
        raise NotImplementedError("2024 version is not implemented")
    else:
        raise ValueError(f"Invalid version: {version}")

    if "train" in splits and anchor_init_config.num_anchor > 0:
        anchor_initialization(
            ann_file=os.path.join(output_dir, "train"),
            num_anchor=anchor_init_config.num_anchor,
            detection_range=anchor_init_config.detection_range,
            sample_ratio=anchor_init_config.sample_ratio,
            output_file_name=os.path.join(output_dir, anchor_init_config.output_file_name),
            verbose=verbose,
        )


def _video_to_frames(video_path: str, image_dir: str, num_frames: int = -1):
    """Decode a synthetic-camera video with the container's supported backend."""
    os.makedirs(image_dir, exist_ok=True)
    existing_frames = sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    required_frames = num_frames if num_frames > 0 else 1
    if len(existing_frames) >= required_frames:
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-c:v", "h264_cuvid",
            "-i", video_path,
        ]
        if num_frames > 0:
            command.extend(["-frames:v", str(num_frames)])
        command.extend(["-start_number", "0", os.path.join(image_dir, "%09d.jpg")])
        subprocess.run(command, check=True)
    elif video2frame_multi_cameras_syn is not None:
        # Compatibility path for older images. New release images deliberately
        # build OpenCV without FFmpeg and provide the audited ffmpeg binary.
        video2frame_multi_cameras_syn(os.path.dirname(os.path.dirname(video_path)))
    else:
        raise RuntimeError(
            "No supported video decoder is available. Install ffmpeg or use a "
            "legacy spatialai-data-utils package with "
            "video2frame_multi_cameras_syn."
        )

    produced_frames = [
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not produced_frames:
        raise RuntimeError(
            f"Video decoder produced no frames for {video_path}. "
            "The Data Services image must include the supported ffmpeg H.264 decoder."
        )


def video_to_frame(root_path: str, splits: list, num_frames: int = -1):
    """Convert AICity videos to frames.

    Args:
        root_path (str): Root path.
        splits (list): Splits.
    """
    for split in splits:
        split_root_path = os.path.join(root_path, split)
        scene_names = sorted(
            name for name in os.listdir(split_root_path)
            if os.path.isdir(os.path.join(split_root_path, name))
        )
        for scene_name in scene_names:
            scene_path = os.path.join(split_root_path, scene_name)
            logger.info(f" Converting video to frames for scene: {scene_path}...")
            assert os.path.isdir(os.path.join(scene_path, "videos")), f"Videos directory not found at {scene_path}"
            videos_dir = os.path.join(scene_path, "videos")
            for video_name in sorted(os.listdir(videos_dir)):
                if not video_name.lower().endswith(".mp4"):
                    continue
                camera_name = os.path.splitext(video_name)[0]
                _video_to_frames(
                    os.path.join(videos_dir, video_name),
                    os.path.join(scene_path, camera_name, "rgb"),
                    num_frames=num_frames,
                )
            logger.info(f" Video to frames conversion completed for scene: {scene_path}")


def load_class_config_from_file(config_path):
    """Load class config from file.

    Args:
        config_path (str): Config path.
    """
    with open(config_path, "r", encoding='utf-8') as f:
        cfg_dict = json.load(f)

    CLASS_MAPPING_DICT = {}
    for cid, c in enumerate(cfg_dict["CLASS_LIST"]):
        CLASS_MAPPING_DICT[c] = cid
    cfg_dict["CLASS_MAPPING_DICT"] = CLASS_MAPPING_DICT

    MAP_SUB_CLASS_TO_CLASS_DICT = {}
    for c in cfg_dict["SUB_CLASS_DICT"].keys():
        for sub_c in cfg_dict["SUB_CLASS_DICT"][c]:
            MAP_SUB_CLASS_TO_CLASS_DICT[sub_c] = c
    cfg_dict["MAP_SUB_CLASS_TO_CLASS_DICT"] = MAP_SUB_CLASS_TO_CLASS_DICT

    return cfg_dict


def update_class_config(cfg_dict):
    """Update class config.

    Args:
        cfg_dict (dict): Class config.
    """
    cfg_dict_new = {}
    for k, v in cfg_dict.items():
        cfg_dict_new[k] = v
    cfg_dict_new["CLASS_MAPPING_DICT"] = {}
    cfg_dict_new["MAP_SUB_CLASS_TO_CLASS_DICT"] = {}

    # Update mapping from CLASS_LIST to class IDs
    for cid, c in enumerate(cfg_dict["CLASS_LIST"]):
        cfg_dict_new["CLASS_MAPPING_DICT"][c] = cid

    # Ensure SUB_CLASS_DICT exists before using it
    if "SUB_CLASS_DICT" in cfg_dict:
        for c in cfg_dict["SUB_CLASS_DICT"].keys():
            for sub_c in cfg_dict["SUB_CLASS_DICT"][c]:
                cfg_dict_new["MAP_SUB_CLASS_TO_CLASS_DICT"][sub_c] = c

    return cfg_dict_new


def create_ov_infos_aicity2025(
    root_path, info_prefix, rgb_format='jpg', depth_format='h5',
    version='2025', split='train', class_config=None,
    camera_group_config=None, recentering=False,
    num_frames=-1,
    verbose=False
):
    """Create info file of AICity2025 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        h5_file (bool): Whether to use h5 file.
        split (str): Split of the data.
        class_config (dict): Class config.
        camera_group_config (dict): Camera group config.
        recentering (bool): Whether to recenter the data.
    """
    split_root_path = os.path.join(root_path, split)
    scene_names = sorted(
        name for name in os.listdir(split_root_path)
        if os.path.isdir(os.path.join(split_root_path, name))
    )

    logger.info(
        f"split_type: {split} | # of scenes: {len(scene_names)} | scene names: {scene_names}"
    )
    os.makedirs(info_prefix, exist_ok=True)

    infos, scene_names_with_grouping = _fill_trainval_infos(
        split_root_path, scene_names, info_prefix,
        rgb_format=rgb_format, depth_format=depth_format,
        load_anno=True, class_config=class_config,
        camera_group_config=camera_group_config,
        recentering=recentering,
        num_frames=num_frames,
        verbose=verbose,
    )

    metadata = {
        "version": version,
        "split_type": split,
        "schema_version": AICITY_OVPKL_SCHEMA_VERSION,
        "class_names": [str(name) for name in class_config["CLASS_LIST"]],
    }
    data = {}
    data["metadata"] = metadata

    if camera_group_config is not None and \
            (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
        scene_names_to_be_processed = scene_names_with_grouping
    else:
        scene_names_to_be_processed = scene_names

    for scene_id, scene_name in enumerate(scene_names_to_be_processed):
        data["infos"] = infos[scene_id]
        info_path = os.path.join(f"{info_prefix}/{split}",
                                 f"{scene_name}_infos_{split}.pkl")
        os.makedirs(f"{info_prefix}/{split}", exist_ok=True)
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"{info_path} saved!")


def _fill_trainval_infos(
    root_path, scene_names, info_prefix,
    rgb_format='jpg', depth_format='h5',
    load_anno=True, class_config=None,
    camera_group_config=None, recentering=False,
    num_frames=-1,
    verbose=False
):
    """Generate the train/val infos from the raw data.

    Args:
        root_path (str): Path of the data root.
        scene_names (list[str]): List of scene names.
        info_prefix (str): Prefix of the info file to be generated.
        h5_file (bool): Whether to use h5 file.
        load_anno (bool): Whether to load annotations.
        camera_group_config (dict): Camera group config.
        recentering (bool): Whether to recenter the data.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    scene_names = sorted(scene_names)
    logger.info(f"scene_names to be processed: {scene_names}")

    infos = []
    sub_scene_names_out = []

    # Dictionary to store global object id.
    object_hash_to_object_id_file = os.path.join(info_prefix, "map_object_hash_to_object_id.json")
    if os.path.exists(object_hash_to_object_id_file):
        with open(object_hash_to_object_id_file, "r", encoding='utf-8') as file:
            map_object_hash_to_object_id = json.load(file)
        if verbose:
            logger.info(f"loaded hashmap from {object_hash_to_object_id_file}")
            logger.info(f"  # of objects: {map_object_hash_to_object_id['no_of_objects']}")
    else:
        map_object_hash_to_object_id = {"no_of_objects": 0, "max_object_id": 0, "objects": {}}

    for scene_name in scene_names:
        scene_path = os.path.join(root_path, scene_name)
        if verbose:
            logger.info(f"scene_path: {scene_path}")
            logger.info("loading calibration ...")
        calib_dict, group_area_dict = load_calib(
            scene_path,
            calib_mode="aic25",
            camera_group_config=camera_group_config,
        )
        if verbose:
            for g in sorted(calib_dict):
                logger.info(f"  # of cameras in {scene_name} {g}: {len(calib_dict[g])} | {list(calib_dict[g].keys())}")

        if load_anno:
            if verbose:
                logger.info("loading ground truth ...")
            with open(os.path.join(scene_path, "ground_truth.json"), "r", encoding='utf-8') as f:
                ground_truths_dict = json.load(f)

        cam_names = sorted(
            get_cam_names_in_scene(scene_path, h5_file=(rgb_format == "h5"))
        )
        if rgb_format == "h5":
            with h5py.File(os.path.join(scene_path, cam_names[0]), "r") as f:
                n_frames = len(f["rgb"])
        else:
            if camera_group_config is not None and \
                    (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                n_frames = len(
                    os.listdir(
                        os.path.join(
                            scene_path,
                            sorted(calib_dict[sorted(calib_dict)[0]])[0],
                            "rgb"
                        )
                    )
                )
            else:
                n_frames = len(
                    os.listdir(
                        os.path.join(
                            scene_path,
                            sorted(calib_dict)[0],
                            "rgb"
                        )
                    )
                )

        # Use the number of frames specified in the config file
        if num_frames != -1:
            n_frames = min(n_frames, num_frames)

        if camera_group_config is not None and \
                (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
            sub_scene_names = []
            for group_name in sorted(calib_dict):
                sub_scene_names.append(f"{scene_name}+{group_name}")
        else:
            sub_scene_names = [scene_name]

        sub_scene_names_out.extend(sub_scene_names)

        for sub_scene_name in sub_scene_names:
            logger.info(f"Converting scene {sub_scene_name} ...")
            train_scene_nusc_infos = []
            if camera_group_config is not None and \
                    (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                group_name = sub_scene_name.split("+")[-1]
            else:
                group_name = None

            # object class statistics
            object_class_set = {}

            for frame_id in tqdm.tqdm(range(n_frames)):
                info = {
                    "frame_idx": frame_id,
                    "cams": {},
                    "scene_name": scene_name,
                    "timestamp": frame_id / FPS,
                    "token": f"{sub_scene_name}__{frame_id:09}",
                }
                if camera_group_config is not None and \
                        (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                    info["scene_name"] = sub_scene_name
                    info["group_name"] = group_name
                    group_origin = [0, 0]
                    if recentering and group_area_dict is not None:
                        # add re-centering for each sub scene
                        if group_name in group_area_dict:
                            group_origin = group_area_dict[group_name]["origin"]

                for cam_name in cam_names:
                    cam = _camera_name_from_storage(cam_name)
                    if camera_group_config is not None and \
                            (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                        if cam not in calib_dict[group_name]:
                            # only consider camera folders according to calibration file
                            continue
                    else:
                        if cam not in calib_dict:
                            # only consider camera folders according to calibration file
                            continue
                    if rgb_format == "h5":
                        cam_path = (os.path.join(scene_path, cam_name), os.path.join("rgb", f"rgb_{frame_id:05}.jpg"))
                        if not os.path.exists(cam_path[0]):
                            logger.warning(f"no file found at {cam_path[0]}")
                    else:
                        cam_path = os.path.join(scene_path, cam, "rgb", f"rgb_{frame_id:05}.{rgb_format}")
                        if not os.path.exists(cam_path):
                            cam_path = os.path.join(scene_path, cam, "rgb", f"{frame_id:09}.{rgb_format}")
                        if not os.path.exists(cam_path):
                            logger.warning(f"no file found at {cam_path}")
                    if load_anno:
                        if depth_format == "h5":
                            dm_path, dm_relative_path = _get_h5_depth_path(
                                scene_path,
                                scene_name,
                                cam_name,
                                frame_id,
                            )
                            if not os.path.exists(dm_path[0]):
                                logger.warning(f"no file found at {dm_path[0]}")
                        else:
                            dm_path = os.path.join(scene_path, cam, "distance_to_image_plane_png", f"distance_to_image_plane_{frame_id:05}.{depth_format}")
                            dm_relative_path = f"{scene_name}/{cam}/distance_to_image_plane_png/distance_to_image_plane_{frame_id:05}.{depth_format}"
                            if not os.path.exists(dm_path):
                                logger.warning(f"no file found at {dm_path}")
                    else:
                        dm_path = ""
                        dm_relative_path = ""
                    sd_token = f"{info['token']}+{cam}"
                    if camera_group_config is not None and \
                            (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                        cam_intrinsic = np.array(calib_dict[group_name][cam]["intrinsic matrix"])
                        cam_sensor2world = np.array(calib_dict[group_name][cam]["projection matrix w2c"])
                        if recentering and group_area_dict is not None:
                            # shift camera to group center
                            cam_world2sensor = np.linalg.inv(cam_sensor2world)
                            cam_world2sensor[0, -1] -= group_origin[0]
                            cam_world2sensor[1, -1] -= group_origin[1]
                            cam_sensor2world = np.linalg.inv(cam_world2sensor)
                    else:
                        cam_intrinsic = np.array(calib_dict[cam]["intrinsic matrix"])
                        cam_sensor2world = np.array(calib_dict[cam]["projection matrix w2c"])

                    cam_info = {}
                    cam_info.update(data_path=cam_path)
                    cam_info.update(depth_map_path=dm_relative_path)
                    cam_info.update(sample_data_token=sd_token)
                    cam_info.update(cam_intrinsic=cam_intrinsic)
                    cam_info.update(sensor2world_transform=cam_sensor2world)
                    info['cams'].update({cam: cam_info})

                # obtain annotation
                if load_anno:
                    if str(frame_id) in ground_truths_dict:
                        annotations = ground_truths_dict[str(frame_id)]
                    else:
                        annotations = []
                    if str(frame_id - 1) in ground_truths_dict:
                        annotations_prev = ground_truths_dict[str(frame_id - 1)]
                    else:
                        annotations_prev = []

                    locs = np.array([anno["3d location"]
                                    for anno in annotations]).reshape(-1, 3)
                    dims = np.array([anno["3d bounding box scale"]
                                    for anno in annotations]).reshape(-1, 3)
                    rots = np.array([- anno["3d bounding box rotation"][2]  # TODO: check object orientation
                                    for anno in annotations]).reshape(-1, 1)

                    # Get global ids from object name
                    instance_global_inds = []
                    for anno in annotations:
                        object_name = anno.get("object name", str(anno["object id"]))  # use track id if object name is not present
                        # If object is not present
                        if object_name not in map_object_hash_to_object_id["objects"]:
                            map_object_hash_to_object_id["max_object_id"] += 1
                            map_object_hash_to_object_id["objects"][object_name] = map_object_hash_to_object_id["max_object_id"]
                            map_object_hash_to_object_id["no_of_objects"] = len(map_object_hash_to_object_id["objects"])
                        instance_global_inds.append(map_object_hash_to_object_id["objects"][object_name])
                    instance_global_inds = np.asarray(
                        instance_global_inds,
                        dtype=np.int64,
                    )

                    instance_inds = np.asarray(
                        [anno["object id"] for anno in annotations],
                        dtype=np.int64,
                    )  # TODO: check if duplicated ids for different object types
                    velocity = _get_object_velocity(locs, instance_inds, annotations_prev)
                    valid_flag = np.array(
                        [True for anno in annotations],
                        dtype=bool).reshape(-1)

                    if recentering and group_area_dict is not None:
                        locs[:, 0] -= group_origin[0]
                        locs[:, 1] -= group_origin[1]

                    # filter by object type
                    names = [anno["object type"] for anno in annotations]
                    names = [class_config["MAP_SUB_CLASS_TO_CLASS_DICT"].get(n, n) for n in names]
                    names = np.asarray(names, dtype=str)
                    keep = [n in class_config["CLASS_LIST"] for n in names]

                    boxes_2d_full = []
                    boxes_2d_visible = []
                    visibilities = []
                    for annotation in annotations:
                        full_value = annotation.get("2d bounding box")
                        visible_value = annotation.get("2d bounding box visible")
                        full_boxes = (
                            {c: np.array(b) for c, b in full_value.items()}
                            if full_value is not None else None
                        )
                        visible_boxes = (
                            {c: np.array(b) for c, b in visible_value.items()}
                            if visible_value is not None else None
                        )
                        boxes_2d_full.append(full_boxes)
                        boxes_2d_visible.append(visible_boxes)

                        if full_boxes is None:
                            visibilities.append(-1)
                            continue
                        visibility_by_camera = {}
                        for camera_name, full_box in full_boxes.items():
                            if visible_boxes is not None and camera_name in visible_boxes:
                                visible_area = _calculate_bbox_area(
                                    visible_boxes[camera_name]
                                )
                                full_area = _calculate_bbox_area(full_box)
                                visibility = (
                                    visible_area / full_area if full_area else 0.0
                                )
                            else:
                                visibility = 0.0
                            visibility_by_camera[camera_name] = visibility
                        visibilities.append(visibility_by_camera)

                    # Initialize has_2d_boxes to prevent UnboundLocalError
                    has_2d_boxes = any(boxes_2d is not None and len(boxes_2d) > 0 for boxes_2d in boxes_2d_visible)

                    # we need to convert box size to
                    # the format of our lidar coordinate system
                    # which is x_size, y_size, z_size (corresponding to l, w, h)
                    gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                    assert len(gt_boxes) == len(
                        annotations
                    ), f"{len(gt_boxes)}, {len(annotations)}"

                    instance_inds = instance_inds[keep]
                    instance_global_inds = instance_global_inds[keep]
                    gt_boxes = gt_boxes[keep]
                    names = names[keep]
                    velocity = velocity[keep]
                    valid_flag = valid_flag[keep]
                    boxes_2d_full = [b for k, b in zip(keep, boxes_2d_full) if k]
                    boxes_2d_visible = [b for k, b in zip(keep, boxes_2d_visible) if k]
                    visibilities = [b for k, b in zip(keep, visibilities) if k]

                    if camera_group_config is not None and (camera_group_config["use_camera_groups"] or camera_group_config["use_training_camera_groups"]):
                        # Re-check if any 2D bounding boxes are available after filtering
                        has_2d_boxes = any(boxes_2d is not None and len(boxes_2d) > 0 for boxes_2d in boxes_2d_visible)
                        if has_2d_boxes:
                            # remove invisible annotations for sub scenes
                            keep = []
                            cams_in_group = list(info["cams"].keys())
                            for bid, boxes_2d in enumerate(boxes_2d_visible):
                                if boxes_2d is not None:
                                    visible_cams_global = list(boxes_2d.keys())
                                    is_box_visible_in_group = any(
                                        x in cams_in_group for x in visible_cams_global
                                    )
                                else:
                                    is_box_visible_in_group = False
                                if is_box_visible_in_group:
                                    keep.append(bid)
                                else:
                                    valid_flag[bid] = False
                            instance_inds = instance_inds[keep]
                            instance_global_inds = instance_global_inds[keep]
                            gt_boxes = gt_boxes[keep]
                            names = names[keep]
                            velocity = velocity[keep]
                            valid_flag = valid_flag[keep]
                            boxes_2d_full = [b for bid, b in enumerate(boxes_2d_full) if bid in keep]
                            boxes_2d_visible = [b for bid, b in enumerate(boxes_2d_visible) if bid in keep]
                            visibilities = [b for bid, b in enumerate(visibilities) if bid in keep]
                        else:
                            logger.debug("No 2D bounding boxes available - skipping camera group filtering, keeping all objects")

                    # get object class statistics
                    for c, ind in zip(names, instance_inds):
                        if c not in object_class_set:
                            object_class_set[c] = {}
                        if ind not in object_class_set[c]:
                            object_class_set[c][ind] = 0
                        else:
                            object_class_set[c][ind] += 1

                    info["instance_inds"] = instance_inds
                    info["asset_inds"] = instance_global_inds
                    info["gt_boxes"] = gt_boxes
                    info["gt_names"] = names
                    info["gt_velocity"] = velocity
                    info["valid_flag"] = valid_flag
                    if has_2d_boxes:
                        info["gt_visibility"] = visibilities

                train_scene_nusc_infos.append(info)

            if verbose:
                # print object class statistics
                logger.info(f"Object statistics in {sub_scene_name}:")
                logger.info(f"  {'object type'.ljust(20)} | # of tracks")
                logger.info(f"  {'-' * 20} | {'-' * 20}")
                object_class_set = dict(sorted(object_class_set.items()))
                for c, v in object_class_set.items():
                    logger.info(f"  {c.ljust(20)} | {len(v)}")
                    logger.info()

            if verbose:
                logger.info(f"append {scene_name} to split train/test set")
            infos.append(train_scene_nusc_infos)

            with open(object_hash_to_object_id_file, 'w', encoding='utf-8') as f:
                json.dump(map_object_hash_to_object_id, f, sort_keys=True)

    return infos, sub_scene_names_out


def _calculate_bbox_area(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    return area


def _get_object_velocity(locations, instance_inds, annotations_prev, vel_dim=3):
    if len(annotations_prev) == 0:
        return np.zeros((len(locations), vel_dim))
    locations_prev = copy.deepcopy(locations)  # copy from current frame: if object not found in previous frame, vel will be zeros
    obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(instance_inds)}
    for anno in annotations_prev:
        obj_id = anno["object id"]
        if obj_id in obj_id_to_idx:
            # obj_id in previous frame can be found
            locations_prev[obj_id_to_idx[obj_id]] = anno["3d location"]
        else:
            # obj_id does not appear in current frame
            pass
    velocities = (locations - locations_prev) * FPS

    # sanity check on velocity
    assert not np.any(np.abs(velocities) > 5), \
        "invalid object velocities\n" \
        f"  object ID {instance_inds[np.where(np.abs(velocities) > 5)[0]]}\n" \
        f"  velocities: {velocities[np.where(np.abs(velocities) > 5)[0]].tolist()}"

    return velocities[:, :vel_dim]


def anchor_initialization(
    ann_file,
    num_anchor=900,
    detection_range=-1,
    sample_ratio=-1,
    output_file_name="kmeans900.npy",
    verbose=False,
):
    """Initialize GT into anchors.

    Args:
        ann_file (str): Annotation file.
        num_anchor (int): Number of anchors.
        detection_range (int): Detection range.
        sample_ratio (int): Sample ratio.
        output_file_name (str): Output file name.
        verbose (bool): Verbosity.
    """
    logger.info(f"Initializing GT into {num_anchor} anchors, detection range {detection_range}, sample ratio {sample_ratio} ...")
    if os.path.isdir(ann_file):
        data = {
            "infos": [],
            "metadata": {}
        }
        ann_files = sorted(
            name for name in os.listdir(ann_file)
            if _is_annotation_pkl(name) and
            os.path.isfile(os.path.join(ann_file, name))
        )
        for ann_idx, scene_name in enumerate(ann_files):
            ann_path = os.path.join(ann_file, scene_name)
            if verbose:
                logger.info(f"[{ann_idx + 1}/{len(ann_files)}] loading annotation from {ann_path} ...")
            with open(ann_path, "rb") as f:
                data_scene = pickle.load(f)
            data["infos"].extend(data_scene["infos"])
            data["metadata"] = data_scene["metadata"]
    else:
        with open(ann_file, "rb") as f:
            data = pickle.load(f)

    box_arrays = [
        np.asarray(info["gt_boxes"])
        for info in data["infos"]
        if info.get("gt_boxes") is not None and np.asarray(info["gt_boxes"]).size
    ]
    gt_boxes = (
        np.concatenate(box_arrays, axis=0)
        if box_arrays else np.zeros((0, 7), dtype=np.float32)
    )
    logger.info(f"Number of ground truth boxes: {len(gt_boxes)}")
    if sample_ratio > 0:
        gt_boxes = gt_boxes[::sample_ratio]
    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    if detection_range > 0:
        mask = distance <= detection_range
        gt_boxes = gt_boxes[mask]

    if gt_boxes.shape[0] == 0:
        logger.warning(
            f"No ground truth boxes available for anchor initialization after filtering. "
            f"Ann file/dir: {ann_file}, detection_range: {detection_range}, sample_ratio: {sample_ratio}. "
            f"Skipping anchor generation."
        )
        return

    current_num_anchor = num_anchor
    if gt_boxes.shape[0] < num_anchor:
        logger.warning(
            f"Number of available ground truth boxes ({gt_boxes.shape[0]}) is less than "
            f"the configured number of anchors ({num_anchor}) for {ann_file}. "
            f"Adjusting number of anchors to {gt_boxes.shape[0]}."
        )
        current_num_anchor = gt_boxes.shape[0]
        # current_num_anchor will be at least 1 here, because gt_boxes.shape[0] > 0

    logger.info("===========Starting kmeans, please wait.===========")
    with threadpool_limits(limits=64, user_api="blas"):
        clf = KMeans(n_clusters=current_num_anchor, verbose=verbose, n_init='auto')
        clf.fit(gt_boxes[:, [X, Y, Z]])
    anchor = np.zeros((current_num_anchor, 11))
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    anchor[:, COS_YAW] = 1
    np.save(output_file_name, anchor)
    logger.info(f"===========Done! Save results to {output_file_name}.===========")
