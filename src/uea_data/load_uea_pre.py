import numpy
# from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
try:
    from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
except ImportError:  # Old sklearn versions
    from sklearn.utils.estimator_checks import NotAnArray
try:
    from scipy.io import arff
    HAS_ARFF = True
except:
    HAS_ARFF = False
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from pyts import datasets


def load_arff_uea(dataset_path):
    """Load arff file for uni/multi variate dataset

    Parameters
    ----------
    dataset_path: string of dataset_path
        Path to the ARFF file to be read

    Returns
    -------
    x: numpy array of shape (n_timeseries, n_timestamps, n_features)
        Time series dataset
    y: numpy array of shape (n_timeseries, )
        Vector of targets

    Raises
    ------
    ImportError: if the version of *Scipy* is too old (pre 1.3.0)
    Exception: on any failure, e.g. if the given file does not exist or is
               corrupted
    """
    if not HAS_ARFF:
        raise ImportError("scipy 1.3.0 or newer is required to load "
                          "time series datasets from arff format.")

    data, meta = arff.loadarff(dataset_path.strip())
    names = meta.names()  # ["input", "class"] for multi-variate

    # firstly get y_train
    y_ = data[names[-1]]  # data["class"]
    y = numpy.array(y_).astype("str")

    # get x_train
    if len(names) == 2:  # len=2 => multi-variate
        x_ = data[names[0]]
        x_ = numpy.asarray(x_.tolist())

        nb_example = x_.shape[0]
        nb_channel = x_.shape[1]
        length_one_channel = len(x_.dtype.descr)
        x = numpy.empty([nb_example, length_one_channel, nb_channel])

        for i in range(length_one_channel):
            # x_.dtype.descr: [('t1', '<f8'), ('t2', '<f8'), ('t3', '<f8')]
            time_stamp = x_.dtype.descr[i][0]  # ["t1", "t2", "t3"]
            x[:, i, :] = x_[time_stamp]

    else:  # uni-variate situation
        x_ = data[names[:-1]]
        x = numpy.asarray(x_.tolist(), dtype=numpy.float32)
        x = x.reshape(len(x), -1, 1)

    return x, y


def load_uea(dataset='Libras'):
    print("Test dataset name = ", dataset)
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    father_path = father_path + "/data_files/"

    datasets.fetch_uea_dataset(dataset, True, father_path, False)

    full_path = os.path.join(father_path, dataset)

    X_train, y_train = load_arff_uea(
        os.path.join(full_path, dataset + "_TRAIN.arff")
    )
    X_test, y_test = load_arff_uea(
        os.path.join(full_path, dataset + "_TEST.arff")
    )

    # X_train, y_train, X_test, y_test = load_txt_uea(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert (y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    if np.isnan(X_scaled).any():
        X_scaled = np.nan_to_num(X_scaled, copy=False, nan=0.0)
    return X_scaled, y