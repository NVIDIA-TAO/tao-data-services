# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import PIL
import PIL
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from joblib import parallel_backend
from joblib import Parallel, delayed
from split_data.utils.util import read_kitti_labels, save_kitti_labels
from pdb import set_trace as bp
from argparse import ArgumentParser
from PIL import Image
from distutils.dir_util import copy_tree
from itertools import chain,repeat

def copy_files(f,dest_dir,splits,split_count,images_path,labels_path):
    if ".txt" in f:
        try:        
            dest_split_dir = dest_dir + "/%d/" % (split_count % splits) 
            image_dest_dir = dest_split_dir + "/images_final_hres"
            label_dest_dir = dest_split_dir + "/labels_final_hres"
            img_id,ext = f.split(".txt")
            img = img_id + ".jpg"
            seq_dir = images_path.split("/")[-3]
            image_file = os.path.join(images_path,img)
            label_file = os.path.join(labels_path,f)
            dest_image_file = os.path.join(image_dest_dir,img)
            dest_label_file = os.path.join(label_dest_dir,f)
            missing_count =0
            if not os.path.getsize(label_file) == 0:
                shutil.copyfile(label_file,dest_label_file)
                shutil.copyfile(image_file,dest_image_file)
            else:
                print(label_file)
        except Exception as e:
            print ("Not a right file")
            print (f)
    else:
        print ("Not a text file")
        print (f)

def sample_dataset(dataset,splits,dest_dir,percent):
    dataset_file =  str("data_sources.csv")
    file_name = os.path.join(dest_dir,dataset_file)
    images_path = dataset + "/images_final_hres/"
    labels_path = dataset + "/labels_final_hres/"
    if os.path.exists(labels_path):
        data = os.listdir(labels_path)
        if len(data) < 10:
            print ("data source is wrong")
            print (dataset)
            images_path = dataset + "/images_final_960x544"
            labels_path = dataset + "/labels_final_960x544/"
            print (images_path)
            print (labels_path)
            data = os.listdir(labels_path)
            print ("other folder works", len(data))
        
        if percent < 1:
            data_samples = round(percent*len(data))
            print ("samples", data_samples)
            data_list = random.sample(data,k=int(data_samples))
        else:
            data_list = list(chain.from_iterable(repeat(item, percent) for item in data))

        file = open(file_name, 'w')
        quick_count = 0
        random.shuffle(data_list)
        for item in data_list:
            split_num = quick_count % splits 
            file.write("%s,%s,%d,%s\n" % (dest_dir,item,split_num,dataset))
            quick_count +=1
        file.close()
               
        split_count = 0
        with parallel_backend("threading", n_jobs=8):
           Parallel(verbose=10)(delayed(copy_files)(f,dest_dir,splits,split_count,images_path,labels_path) for split_count,f in enumerate(data_list))

    else: 
        print ("labels_dir does not exist")


def create_dir_if_not_exists(dir):
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

def copy_dataset(project_folder,rotation_folder,item):
    image_id = item[1].split(".txt")[0]
    src_image_path = project_folder + "/images_final_fres/" + image_id + ".jpg"
    src_labels_path = project_folder + "/labels_final_fres/" + image_id + ".txt"
    dest_image_folder = rotation_folder  + str(item[2]) + "/images_final_fres/" 
    dest_labels_folder = rotation_folder  + str(item[2]) + "/labels_final_fres/" 
    ensureDir(dest_image_folder)
    ensureDir(dest_labels_folder)
    dest_image_path = dest_image_folder + image_id + ".jpg"
    dest_labels_path = dest_labels_folder + image_id + ".txt"
    try:
        shutil.copyfile(src_image_path,dest_image_path)
        shutil.copyfile(src_labels_path,dest_labels_path)
    except:
        print (src_image_path)


def split(inp_dir, dest_dir, n=12, p=1):
    create_dir_if_not_exists(dest_dir)
    for i in range(n):
        dest_split_dir = dest_dir + "/%d/" % (i) 
        create_dir_if_not_exists(dest_split_dir)
        image_dest_dir = dest_split_dir + "/images_final_hres"
        create_dir_if_not_exists(image_dest_dir)
        label_dest_dir = dest_split_dir + "/labels_final_hres"
        create_dir_if_not_exists(label_dest_dir)

    sample_dataset(inp_dir,n,dest_dir,p)

def parse_command_line(args=None):
    """
    Function to parse the command line arguments

    Args:
        args(list): list of arguments to be parsed.
    """
    parser = argparse.ArgumentParser(
        prog='split_data',
        description='Split the dataset for performing augmentation'
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        required=True,
        help='Snapshot data directory'
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        required=True,
        help='Directory to output the data splits'
    )
    parser.add_argument(
        '-n',
        '--splits',
        type=int,
        required=True,
        help='Number of splits'
    )
    return parser.parse_args(args)

def main(cl_args=None):
    """
    Split dataset and perform augmentations.

    Args:
        cl_args(list): list of arguments.
    """
    args = parse_command_line(args=cl_args)
    try:
        split([args.input_directory], args.output_directory, args.splits)
    except RuntimeError as e:
        print("Data Split execution failed with error: {}".format(e))
        exit(-1)


if __name__ == '__main__':
    main()