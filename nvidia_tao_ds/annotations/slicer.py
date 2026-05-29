# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Annotation Slicer."""
from abc import ABC, abstractmethod
import random
import re
import os
import json
from pycocotools.coco import COCO
from nvidia_tao_ds.core.logging.logging import logger


class Slicer(ABC):
    """Base Slicer class."""

    def __init__(self, annotation_file):
        """Initialize."""
        self.annotation_file = annotation_file

    def update(self, annotation_file):
        """Update annotation file."""
        self.annotation_file = annotation_file

    @abstractmethod
    def slice(self, output_dir, filter_config):
        """Slice."""
        pass


def dump_json(json_dict, filename):
    """Helper function to dump json file."""
    logger.info(f"Writing to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


class COCORandomSlicer(Slicer):
    """Random split COCO slicer."""

    def __init__(self, annotation_file):
        """Init."""
        super().__init__(annotation_file)
        assert isinstance(annotation_file, str)

    def slice(self, output_dir, filter_config):
        """Slice the annotation to generate random portion or uniform split."""
        random.seed(42)
        logger.info(f'Loading {self.annotation_file}')
        data = COCO(self.annotation_file)
        split = filter_config.split
        # TODO(@yuw): enable cat remap
        assert filter_config.reuse_categories, "Feature not available yet!"
        output = {}
        output['categories'] = data.dataset['categories']

        if isinstance(split, float):
            assert 0 < split < 1, "`split` must be (0, 1) when it is a float"
            num_images = int(filter_config.split * len(data.dataset['images']))
            assert 0 < num_images, "Please verify the split!"

            images = random.sample(data.dataset['images'], num_images)
            output['images'] = images
            image_ids = [image['id'] for image in images]
            annot_ids = data.getAnnIds(imgIds=image_ids, iscrowd=None)
            output['annotations'] = data.loadAnns(annot_ids)
            dump_json(output, os.path.join(output_dir, 'kept.json'))

            if filter_config.dump_remainder:
                image_ids_re = list(set(data.getImgIds()) - set(image_ids))
                output['images'] = data.loadImgs(image_ids_re)
                annot_ids_re = data.getAnnIds(imgIds=image_ids_re, iscrowd=None)
                output['annotations'] = data.loadAnns(annot_ids_re)
                dump_json(output, os.path.join(output_dir, 'remainder.json'))

        if isinstance(split, int):
            assert split > 1, "`split` must be > 1 when it is an integer"
            image_ids = data.getImgIds()
            random.shuffle(image_ids)
            num_samples = (len(image_ids) + split - 1) // split
            for j, i in enumerate(range(0, len(image_ids), num_samples)):
                partition = image_ids[i: i + num_samples]
                output['images'] = data.loadImgs(partition)
                annot_ids = data.getAnnIds(imgIds=partition, iscrowd=None)
                output['annotations'] = data.loadAnns(annot_ids)
                dump_json(output, os.path.join(output_dir, f'part_{j}.json'))


class COCONumberSlicer(Slicer):
    """Number of samples based COCO slicer."""

    def __init__(self, annotation_file):
        """Init."""
        super().__init__(annotation_file)
        assert isinstance(annotation_file, str)

    def slice(self, output_dir, filter_config):
        """Get the slice of the first N samples."""
        logger.info(f'Loading {self.annotation_file}')
        data = COCO(self.annotation_file)
        num_images = filter_config.num_samples
        assert not filter_config.dump_remainder, "Feature not enable yet!"
        assert filter_config.reuse_categories, "Feature not enable yet!"

        output = {}
        output['categories'] = data.dataset['categories']
        images = data.dataset['images'][:num_images]
        output['images'] = images
        image_ids = [image['id'] for image in images]
        annot_ids = data.getAnnIds(imgIds=image_ids, iscrowd=None)
        output['annotations'] = data.loadAnns(annot_ids)
        dump_json(output, os.path.join(output_dir, 'kept.json'))


class COCOCategorySlicer(Slicer):
    """Category based COCO slicer."""

    def __init__(self, annotation_file):
        """Init."""
        super().__init__(annotation_file)
        assert isinstance(annotation_file, str)

    def slice(self, output_dir, filter_config):
        """Slice by including or excluding certain categories."""
        logger.info(f'Loading {self.annotation_file}')
        data = COCO(self.annotation_file)
        categories = data.dataset['categories']
        cat_id_map = {cat['name']: cat['id'] for cat in categories}

        included = set(filter_config.included_categories) or set(cat_id_map.keys())
        excluded = set(filter_config.excluded_categories)
        query_categories = list(included - excluded)
        query_cat_ids = [cat_id_map[cat] for cat in query_categories]

        # Update the category id to start from 1
        remapped_categories, mapping = [], {}
        idx = 1
        for cat in categories:
            if cat['id'] in query_cat_ids:
                mapping[cat['id']] = idx
                cat['id'] = idx
                remapped_categories.append(cat)
                idx += 1

        assert not filter_config.dump_remainder, "Feature not enable yet!"
        assert filter_config.reuse_categories, "Feature not enable yet!"
        image_ids = set()
        for cat_id in query_cat_ids:
            image_ids.update(data.getImgIds(catIds=[cat_id]))

        # categories
        image_ids = list(image_ids)
        output = {}
        output['categories'] = remapped_categories
        output['images'] = data.loadImgs(ids=image_ids)
        annot_ids = data.getAnnIds(imgIds=image_ids, iscrowd=None)
        output['annotations'] = []
        for annot_id in annot_ids:
            annot = data.loadAnns(annot_id)[0]
            if annot['category_id'] in query_cat_ids:
                # update category id to the remapped version
                annot['category_id'] = mapping[annot['category_id']]
                output['annotations'].append(annot)
        dump_json(output, os.path.join(output_dir, 'kept.json'))


class COCOFilenameSlicer(Slicer):
    """Category based COCO slicer."""

    def __init__(self, annotation_file):
        """Init."""
        super().__init__(annotation_file)
        assert isinstance(annotation_file, str)

    def slice(self, output_dir, filter_config):
        """Slice by including or excluding certain categories."""
        assert filter_config.re_patterns, "re_patterns is not specified."
        logger.info(f'Loading {self.annotation_file}')
        data = COCO(self.annotation_file)
        categories = data.dataset['categories']

        output = {}
        output['categories'] = categories
        images, image_ids = [], []
        for image in data.dataset['images']:
            if any(re.match(pattern, image['file_name']) for pattern in filter_config.re_patterns):
                images.append(image)
                image_ids.append(image['id'])
        output['images'] = images
        annot_ids = data.getAnnIds(imgIds=image_ids, iscrowd=None)
        output['annotations'] = data.loadAnns(annot_ids)
        dump_json(output, os.path.join(output_dir, 'kept.json'))


builder_dict = {
    'category': COCOCategorySlicer,
    'number': COCONumberSlicer,
    'random': COCORandomSlicer,
    'filename': COCOFilenameSlicer
}


def builder(config):
    """COCO slicer builder."""
    logger.info(f"Building {config.filter.mode} based COCO slicer.")
    slicer_class = builder_dict[config.filter.mode]
    return slicer_class(config.data.annotation_file)
