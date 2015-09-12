from __future__ import division
from __future__ import print_function
__author__ = 's'


import h5py
import os
import numpy
from logging_tools import setup_logging
import argparse
import logging

import sklearn.dummy
import sklearn.cross_validation
import sklearn.svm
import sklearn.grid_search
import sklearn.metrics
import sklearn.linear_model
import math

def train_classifier(args):
    logger = logging.getLogger(__name__)
    path_data = args['path_data']

    with h5py.File(path_data, "r") as f:
        X_loaded, y_loaded = numpy.array(f['X']),  numpy.array(f['y'])
    logger.info("loaded {}. shape {}".format(path_data, X_loaded.shape))


    limit = 100000
    X = X_loaded[:limit]
    y = y_loaded[:limit].astype('bool')
    logger.info("X.shape {}".format(X.shape))
    ratio = numpy.count_nonzero(y==1)/numpy.count_nonzero(y==0)
    logger.info("ratio = {}, count non zero = {}".format(ratio,numpy.count_nonzero(y==1)))


    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    tuned_parameters = [{'C': [0.01, 0.1,  1, 10, 100, 1000 ]}]
    svc = sklearn.svm.LinearSVC(class_weight={0:1, 1:math.ceil(1/ratio)})
    cv = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=5, random_state=0)

    clf = sklearn.grid_search.GridSearchCV(svc, tuned_parameters, cv=cv, n_jobs=10, verbose=20,
                                       scoring=sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score))

    clf.fit(X_train, y_train)

    # for comparison
    dummy = sklearn.dummy.DummyClassifier(random_state=0)
    dummy.fit(X_train, y_train)

    def test_metric(metric):
        scorer = sklearn.metrics.make_scorer(metric)
        print ()
        print ("using metric {}".format(metric))
        print ("dummy: ", scorer(dummy, X_test, y_test))
        print ("clf:", scorer(clf, X_test, y_test))

    test_metric(sklearn.metrics.accuracy_score)
    test_metric(sklearn.metrics.roc_auc_score)
    print ()
    print ("accuracy must be good enough for dummy classifier, as classes are unbalanced")
    print ("AUC must be much better for trained classifier")



if __name__ == '__main__':
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__,
                                 epilog= \
                                 """Example of usage:\n
    python ./generate_train_set.py -ti "./vid/frame_{:05d}.jpg" -tl "./vid/frame_{:05d}.jpg.labels.txt"
    -d "saved_data_test.hdf5" -s 350 -e 355 -i 1""")

    ap.add_argument("-i", dest='path_data', required=True, help="path to input HDF5 dataset")
    ap.add_argument("-v", dest='verbose', required=False, type=int, default=False, help="verbose output")

    args = vars(ap.parse_args())

    train_classifier(args)



