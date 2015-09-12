#!/usr/bin/python
"""
Training set generator
"""
from __future__ import division
import matplotlib.pyplot as plt


from logging_tools import setup_logging
import logging
import common_tools
import h5py
import os
import numpy
import argparse

plt.ion()

plt.rcParams[ u'image.cmap'] = 'gray'
__author__ = 'Sergey Matyunin'

def generate_train_set(args):
    logger = logging.getLogger(__name__)
    logger_silent = logging.getLogger("silient")
    if not args["verbose"]:
        logger_silent.setLevel(logging.ERROR)

    X_all = []
    y_all = []

    path_templ, path_templ_labels = args['path_templ'], args['path_templ_labels']
    interactive = args["interactive"]
    figure = False
    if interactive:
        figure = plt.figure()

    for idx in range(args['idx_start'], args['idx_end']):

        logger.info("Processing frame {}".format(idx))

        X, y = common_tools.process_one_frame(
            path_templ.format(idx-1),
            path_templ_labels.format(idx-1),
            path_templ.format(idx),
            path_templ_labels.format(idx),
            figure=figure, logger=logger_silent)
        if interactive:
            plt.draw()
            raw_input('Press enter...')

        X_all.extend(X)
        y_all.extend(y)

    X = numpy.array(X_all)
    y = numpy.array(y_all)

    assert(len(X) == len(y))
    logger.info("Processing finished. len(X) = {}".format(len(X)))

    # "saved_X_y_pairs_with_sizes.hdf5"
    path_save = args['path_save']
    if not os.path.exists(path_save):
        with h5py.File(path_save, "w") as f:
            f['X'] = X
            f['y'] = y
        logger.info("saved to {}".format(path_save))


if __name__ == '__main__':
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__,
                                 epilog= \
                                 """Example of usage:\n
    python ./generate_train_set.py -ti "./vid/frame_{:05d}.jpg" -tl "./vid/frame_{:05d}.jpg.labels.txt"
    -d "saved_data_test.hdf5" -s 350 -e 355 -i 1"""
)
    ap.add_argument("-ti", dest='path_templ', required=True, help="Path template for images")
    ap.add_argument("-tl", dest='path_templ_labels', required=True, help="Path template for labels")
    ap.add_argument("-d", dest='path_save', required=True, help="path to destination HDF5 dataset")
    ap.add_argument("-s", dest='idx_start', required=True, type=int, help="index of first image")
    ap.add_argument("-e", dest='idx_end', required=True, type=int, help="index of last image (not included)")
    ap.add_argument("-v", dest='verbose', required=False, type=int, default=False, help="verbose output")
    ap.add_argument("-i", dest='interactive', required=False, type=int, default=False, help="interactive")

    args = vars(ap.parse_args())

    generate_train_set(args)

