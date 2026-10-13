#!/usr/bin/env python3

# standard lib
import sys
import argparse

# numpy family
import numpy as np

# 3rd party
import skimage
from skimage import io
from PIL import Image

# local
sys.path.insert(0, "../pybuild")
sys.path.insert(0, "pybuild")
import boruvka_superpixel

def boruvkasupix(infile, outfile, n_supix):
    img_in = io.imread(infile)
    img_edge = np.zeros((img_in.shape[:2]), dtype=img_in.dtype)
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_2d(img_in, img_edge)

    out = bosupix.label(n_supix);
    image = Image.fromarray(out);
    
    image.save(outfile);

def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile',
            help='input image file')
    parser.add_argument('outfile',
            help='output image file')
    parser.add_argument('n_supix',
            type=int,
            help='number of superpixels')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_arguments(sys.argv[1:])
    boruvkasupix(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
