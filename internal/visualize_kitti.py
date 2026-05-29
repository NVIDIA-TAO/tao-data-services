# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KITTI visualization tool."""
from PIL import Image, ImageDraw
from glob import glob

import os
import argparse


def draw_one_image(image_file, kitti_file, output_dir):
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    f = open(kitti_file, 'r')
    for line in f:
        po = list(map(lambda x:float(x), line.split(' ')[4:8]))
        # print label
        draw.rectangle(po, outline=(0,0,255,200))

    img.save(os.path.join(output_dir, os.path.basename(kitti_file)+'.jpg'))


def draw_dir(imagedir, labeldir, output_dir, ext='.png'):
    assert os.path.isdir(imagedir), "Image dir invalid."
    assert os.path.isdir(labeldir), "Label dir invalid."

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for labelpath in sorted(glob(labeldir + '/*.txt')):
        imgpath = os.path.join(imagedir, os.path.basename(labelpath)[:-4] + ext)
        
        if os.path.isfile(imgpath):
            # print imgpath
            draw_one_image(imgpath, labelpath, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KITTI visualization tool.")
    parser.add_argument('-i', '--imagedir',
                        help="Image Dir",
                        type=str,
                        required=True)
    parser.add_argument('-l', '--labeldir',
                        help="Label Dir",
                        type=str,
                        required=True)
    parser.add_argument('-o', '--outputdir',
                        help="Output Dir",
                        type=str,
                        required=True)
    parser.add_argument('-e', '--extension',
                        help="File extension",
                        type=str,
                        default='.png')
    args = parser.parse_args()
    draw_dir(args.imagedir, args.labeldir, args.outputdir, args.extension)
