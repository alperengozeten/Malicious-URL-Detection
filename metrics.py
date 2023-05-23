import numpy as np

# mean squared error
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.reshape((len(y_true), 1))
    return np.sum(np.square(y_true - y_pred)) / len(y_true)

# mean absolute error
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.reshape((len(y_true), 1))
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]
    return r ** 2