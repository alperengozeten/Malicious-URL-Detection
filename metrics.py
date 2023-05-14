import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error given two vectors
    :param y_true: the true targets
    :param y_pred: the predicted targets
    :return: the mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.square(y_true - y_pred)) / len(y_true)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean absolute error given two vectors
    :param y_true: the true targets
    :param y_pred: the predicted targets
    :return: the mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean absolute percentage error given two vectors
    :param y_true: the true targets
    :param y_pred: the predicted targets
    :return: the mean absolute percentage error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(np.divide(y_true - y_pred, y_true))) / len(y_pred)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the r2 score give two vectors
    :param y_true: the true targets
    :param y_pred: the predicted targets
    :return: the coefficient of determination
    """
    r = np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]
    return r ** 2