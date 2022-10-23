import numpy
# from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
try:
    from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
except ImportError:  # Old sklearn versions
    from sklearn.utils.estimator_checks import NotAnArray
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from pyts import datasets


def to_time_series(ts, remove_nans=False):
    """Transforms a time series so that it fits the format used in ``tslearn``
    models.

    Parameters
    ----------
    ts : array-like
        The time series to be transformed.
    remove_nans : bool (default: False)
        Whether trailing NaNs at the end of the time series should be removed
        or not

    Returns
    -------
    numpy.ndarray of shape (sz, d)
        The transformed time series. This is always guaraneteed to be a new
        time series and never just a view into the old one.

    Examples
    --------
    >>> to_time_series([1, 2])
    array([[1.],
           [2.]])
    >>> to_time_series([1, 2, numpy.nan])
    array([[ 1.],
           [ 2.],
           [nan]])
    >>> to_time_series([1, 2, numpy.nan], remove_nans=True)
    array([[1.],
           [2.]])

    See Also
    --------
    to_time_series_dataset : Transforms a dataset of time series
    """
    ts_out = numpy.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out


def ts_size(ts):
    """Returns actual time series size.

    Final timesteps that have `NaN` values for all dimensions will be removed
    from the count. Infinity and negative infinity ar considered valid time
    series values.

    Parameters
    ----------
    ts : array-like
        A time series.

    Returns
    -------
    int
        Actual size of the time series.

    Examples
    --------
    >>> ts_size([1, 2, 3, numpy.nan])
    3
    >>> ts_size([1, numpy.nan])
    1
    >>> ts_size([numpy.nan])
    0
    >>> ts_size([[1, 2],
    ...          [2, 3],
    ...          [3, 4],
    ...          [numpy.nan, 2],
    ...          [numpy.nan, numpy.nan]])
    4
    >>> ts_size([numpy.nan, 3, numpy.inf, numpy.nan])
    3
    """
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and numpy.all(numpy.isnan(ts_[sz - 1])):
        sz -= 1
    return sz


def to_time_series_dataset(dataset, dtype=numpy.float):
    """Transforms a time series dataset so that it fits the format used in
    ``tslearn`` models.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed. A single time series will
        be automatically wrapped into a dataset with a single entry.
    dtype : data type (default: numpy.float)
        Data type for the returned dataset.

    Returns
    -------
    numpy.ndarray of shape (n_ts, sz, d)
        The transformed dataset of time series.

    Examples
    --------
    >>> to_time_series_dataset([[1, 2]])
    array([[[1.],
            [2.]]])
    >>> to_time_series_dataset([1, 2])
    array([[[1.],
            [2.]]])
    >>> to_time_series_dataset([[1, 2], [1, 4, 3]])
    array([[[ 1.],
            [ 2.],
            [nan]],
    <BLANKLINE>
           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> to_time_series_dataset([]).shape
    (0, 0, 0)

    See Also
    --------
    to_time_series : Transforms a single time series
    """
    try:
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            return to_time_series_dataset(numpy.array(dataset))
    except ImportError:
        pass
    if isinstance(dataset, NotAnArray):  # Patch to pass sklearn tests
        return to_time_series_dataset(numpy.array(dataset))
    if len(dataset) == 0:
        return numpy.zeros((0, 0, 0))
    if numpy.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts, remove_nans=True))
                  for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = numpy.zeros((n_ts, max_sz, d), dtype=dtype) + numpy.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out.astype(dtype)


def load_txt_uea(dataset_path):
    """Load arff file for uni/multi variate dataset

    Parameters
    ----------
    dataset_path: string of dataset_path
        Path to the TXT file to be read

    Returns
    -------
    x: numpy array of shape (n_timeseries, n_timestamps, n_features)
        Time series dataset
    y: numpy array of shape (n_timeseries, )
        Vector of targets

    Raises
    ------
    Exception: on any failure, e.g. if the given file does not exist or is
               corrupted
    """
    data = numpy.loadtxt(dataset_path)
    X = to_time_series_dataset(data[:, 1:])
    y = data[:, 0].astype(numpy.int)
    return X, y


def load_ucr(dataset='CBF'):
    print("Test dataset name = ", dataset)
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    father_path = father_path + "/data_files/"

    datasets.fetch_ucr_dataset(dataset,True,father_path,False)

    full_path = os.path.join(father_path, dataset)

    X_train, y_train = load_txt_uea(
        os.path.join(full_path, dataset + "_TRAIN.txt")
    )
    X_test, y_test = load_txt_uea(
        os.path.join(full_path, dataset + "_TEST.txt")
    )

    # X_train, y_train, X_test, y_test = load_txt_uea(dataset)
    if np.isnan(X_train).any():
        X_train = np.nan_to_num(X_train, copy=False, nan=0.0)
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, copy=False, nan=0.0)
    min_shape1 = min(X_train.shape[1],X_test.shape[1])
    if X_train.shape[1]!=X_test.shape[1]:
        X_train = X_train[:, :min_shape1, :]
        X_test = X_test[:, :min_shape1, :]
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

def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set

