# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import argparse

from time import sleep
from tqdm import tqdm
from tqdm import trange


label_map = {'baggage': ['handbag', 'suitcase', 'backpack'],
         'person': ['person'],
         'lp': ['lp', 'LPD'],
         'face': 'face'}

OMIT_LIST = ['crops']


def read_kitti_labels(label_file):
    "Function wrapper to read kitti format labels txt file."
    label_list = []
    if not os.path.exists(label_file):
        raise ValueError("Labelfile : {} does not exist".format(label_file))
    with open(label_file, 'r') as lf:
        for row in csv.reader(lf, delimiter=' '):
            label_list.append(row)
    lf.closed
    return label_list


def filter_objects(child_objects, class_key):
    "Extract object metadata of class name class key."
    filtered_objects = []
    for item in child_objects:
        if item[0] in label_map[class_key]:
            item[0] = class_key
            # if class_key == 'lp':
            #     item.append("1ABC2345")
            filtered_objects.append(item)
    return filtered_objects


def save_kitti_labels(objects, output_file):
    "Function wrapper to save kitti format bbox labels to txt file."
    with open(output_file, 'w') as outfile:
        outwrite=csv.writer(outfile, delimiter=' ')
        for row in objects:
            outwrite.writerow(row)


def main_kitti_concatenate(args):
    "Function wrapper to concatenate kitti format labels from two different directories for same images to one file."
    class_key = args.filter_class
    parent_root = args.parent_dir
    child_root = args.child_dir
    output_root = args.output_dir
    if not os.path.exists(parent_root):
        raise ValueError("Parent label path: {} does not exist".format(parent_root))
    if not os.path.exists(child_root):
        raise ValueError("Child label path: {} does not exist".format(parent_root))

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    parent_label_list = [ item for item in sorted(os.listdir(parent_root)) if item.endswith('.txt') ]
    for label in tqdm(parent_label_list):
        child_label = os.path.join(child_root, label)
        parent_label = os.path.join(parent_root, label)
        parent_objects = read_kitti_labels(parent_label)
        if os.path.exists(child_label):
            child_objects = read_kitti_labels(child_label)
            filtered_objects = filter_objects(child_objects, class_key)
            if not filtered_objects is None:
                for elem in filtered_objects:
                    parent_objects.append(elem)
        output_file = os.path.join(output_root, label)
        save_kitti_labels(parent_objects, output_file)

def class_rename(input_dir, filter_class, new_classname, output_dir):
    "Function wrapper to rename classes for objects in kitti format label file."
    class_key = filter_class
    parent_root = input_dir
    output_dir = output_dir
    new_name = new_classname
    lv = True
    ns = True

    if not new_name is None:
        if not os.path.exists(parent_root):
            raise ValueError("Input directory: {} does not exist".format(parent_root))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        labels_list = [item for item in os.listdir(parent_root) if item.endswith('.txt')]

        for i in trange(len(labels_list), desc='frame_list', leave=lv):
            renamed_data = []
            label = labels_list[i]
            read_file = os.path.join(parent_root, label)
            output_file = os.path.join(output_dir, label)
            label_data = read_kitti_labels(read_file)

            for item in label_data:
                if item[0].lower() == class_key.lower():
                    item[0] = new_name
                renamed_data.append(item)

            save_kitti_labels(renamed_data, output_file)


def KPI_rename_wrapper(args):
    "Function wrapper to rename label files for multiple sequences in a directory."
    KPI_root = args.parent_dir
    class_key = args.filter_class
    output_root = KPI_root
    new_name = args.new_name

    KPI_directory_list = [os.path.join(KPI_root, item) for item in os.listdir(KPI_root) \
                          if os.path.isdir(os.path.join(KPI_root, item)) and \
                          item not in OMIT_LIST]


    for i in trange(len(KPI_directory_list), desc='sequence_list', leave=True):
        item = KPI_directory_list[i]
        sequence = os.path.join(KPI_directory_list, item)
        input_dir = os.path.join(sequence, 'labels')
        output_dir = os.path.join(sequence, 'labels_renamed')
        class_rename(input_dir, class_key, new_name, output_dir)


def parse_command_line():
    parser = argparse.ArgumentParser(description='Label concatenation tool')
    parser.add_argument('-p',
                        '--parent_dir',
                        help='Parent directory of labels.',
                        default=None)
    parser.add_argument('-ch',
                        '--child_dir',
                        help='Child directory of labels.',
                        default=None)
    parser.add_argument('-o',
                        '--output_dir',
                        help='Output directory for concatenated labels',
                        default=None)
    parser.add_argument('-fc',
                        '--filter_class',
                        help='Key to filter class in child directory',
                        default='None')
    parser.add_argument('-nn',
                        '--new_name',
                        help='Name of the filters class to be renamed',
                        default='None')

    arguments = parser.parse_args()
    return arguments


if __name__=='__main__':
    arguments = parse_command_line()
    KPI_rename_wrapper(arguments)