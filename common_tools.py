#!/usr/bin/python
from __future__ import division
import matplotlib.pyplot as plt
import cv2
import label_img
import skimage.measure
import numpy as np
import numpy
import skimage.morphology
import matplotlib.patches as mpatches
import logging_tools, logging

__author__ = 'Sergey Matyunin'

th_color = 5
th_min_size = 50
th_max_size = 1000000
radius_markup_expansion = 10

def read_frame(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to read {}".format(path))
    return img


def read_markup(path_labels):
    list_rect_xy = label_img.ImageWithROI.load_csv(path_labels)
    return list_rect_xy


def get_segm(diff):
    markers = np.zeros_like(diff)

    markers[diff > th_color] = 2
    markers[diff < -th_color] = 3

    L = skimage.measure.label(markers)
    L = skimage.morphology.remove_small_objects(L, th_min_size)
    return L


def inside(inner, outer):
    """
    Check if inner rectangle is inside outer.
    Format of the rectangle: (x1, y1, x2, y2).

    >>> inside((0, 0, 10, 10), (-1,-1, 9,11))
    False

    >>> inside((0, 0, 10, 10), (-1,-1, 11,11))
    True

    >>> inside((0, 0, 1, 1), (-1,-1, 11,11))
    True

    :param inner:
    :param outter:
    :return: True or False
    """
    return outer[0] <= inner[0] <= outer[2] and \
           outer[0] <= inner[2] <= outer[2] and \
           outer[1] <= inner[1] <= outer[3] and \
           outer[1] <= inner[3] <= outer[3]


def ext_rect(rect, border=0):
    minr, minc, maxr, maxc = rect
    return minr-border, minc-border, maxr+border, maxc+border


def is_in_markup(region, markup1):
    if region.area < th_min_size or region.area > th_max_size:
        return False

    minr, minc, maxr, maxc = region['bbox']

    for rect_gt in markup1:
        if inside((minc, minr, maxc, maxr), rect_gt) and region.mean_intensity > 0:
            return True

    return False


def draw_region(region, color, ax=None):
    """
    draw rectangle around segmented coins
    :param region:
    :param color:
    :return:
    """
    minr, minc, maxr, maxc = region['bbox']
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor=color, linewidth=1)
    if ax is None:
        ax = plt.gca()
    ax.add_patch(rect)


def regions_to_feats(region0, region1,  markup0, markup1, classifier=None):
    step = numpy.array(region0.centroid) - numpy.array(region1.centroid)
    min_area = min(region0.area, region1.area)
    max_area = max(region0.area, region1.area)
    rel_dist = numpy.linalg.norm(step) / (min_area+1)
    feats = numpy.array([
            region0.area / (min_area+1),
            rel_dist,
            region0.mean_intensity,
            region1.mean_intensity,
            min_area,
            max_area,

        ], dtype=numpy.float32)

    if classifier is None:
        positive = is_in_markup(region0, markup0) and is_in_markup(region1, markup1) and rel_dist < 0.8
    else:
        positive = classifier.predict(feats)
    return feats, positive


def read_frame_and_labels(path_img, path_labels):
    return read_frame(path_img), [ext_rect(rect, radius_markup_expansion) for rect in read_markup(path_labels)]


def process_one_frame(path_img0, path_labels0, path_img1, path_labels1,
                      figure=False,
                      logger=logging.getLogger("process_one_frame"),
                      classifier=None):
    """
    Generates features for manual markup (with minor filtering of results) or classifies using trained classifier

    :param path_img0:
    :param path_labels0:
    :param path_img1:
    :param path_labels1:
    :param figure: False for no display, None for new figure, otherwise figure to plot on
    :param logger:
    :param classifier: None or trained classifier
    :return:
    """

    logger.info("Processing frames {}, {}".format(path_img0, path_img1))

    img0, markup0 = read_frame_and_labels(path_img0, path_labels0)
    img1, markup1 = read_frame_and_labels(path_img1, path_labels1)

    diff1 = img1-img0.astype('float32')
    diff0 = -diff1

    L0 = get_segm(diff0)
    L1 = get_segm(diff1)

    if figure is None:
        figure = plt.figure()
    if figure:
        ax = figure.add_subplot(111)
        ax.imshow(L1, cmap=plt.cm.spectral)

    X = []
    y = []

    for region0 in skimage.measure.regionprops(L0, intensity_image=diff0):
        for region1 in skimage.measure.regionprops(L1, intensity_image=diff1):
            feats, positive = regions_to_feats(region0, region1, markup0, markup1, classifier)
            X.append(feats)
            y.append(int(positive))

            logger.info("Positive {}; feats {}".format(int(positive), feats))
            if positive:
                if figure:
                    draw_region(region1, 'red', ax)
                    draw_region(region0, 'white', ax)

    if figure:
        for rect_gt in markup1:
            (minc, minr, maxc, maxr) = rect_gt
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=1)
            ax.add_patch(rect)
    return X, y