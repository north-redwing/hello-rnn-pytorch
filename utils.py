import numpy as np
import torch
import matplotlib.pyplot as plt


def create_dataset(time_series, look_back=1):
    """
    create [look_back] length array from [time-series] data.
    e.g.) ts:{1,2,3,4,5}, lb=3 => {1,2,3},{2,3,4},{3,4,5}.
    """
    sub_seq, nxt = [], []
    for i in range(len(time_series) - look_back):
        sub_seq.append(time_series[i:i + look_back])
        nxt.append(time_series[i + look_back])
    return sub_seq, nxt


def create_sin_curve(look_back):
    """making a Sine curve data."""
    x = np.linspace(-10, 10, 200)
    y = np.sin(x)
    sub_seq, nxt = create_dataset(y, look_back=look_back)
    return sub_seq, nxt, x, y


def split_data(X, y, train_retio=.5):
    """split data into train and test set."""
    train_size = int(len(y) * train_retio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
    return X_train, y_train, X_test, y_test


def get_data(look_back=5):
    sub_seq, nxt, x, y = create_sin_curve(look_back=look_back)
    X_train, y_train, X_test, y_test = split_data(sub_seq, nxt)
    return X_train, y_train, X_test, y_test, x, y


def get_pred_curve(X_test, test_size, model, look_back):
    """input the predicted value to predict the next observation value."""
    pred_curve = []
    tensor_now = X_test[0]
    model.eval()
    with torch.no_grad():
        for _ in range(test_size):
            tensor_now = tensor_now.reshape(1, 1, look_back).float()
            pred = model(tensor_now)
            pred_curve.append(pred.item())
            tensor_now = torch.cat([tensor_now[:, :, 1:], pred], dim=2)
    return pred_curve


def draw_pred_curve(train_size, look_back, x, y, pred_curve, e_num):
    plt.plot(x[:train_size + look_back], y[:train_size + look_back],
             color='blue')
    plt.plot(x[train_size + look_back:], pred_curve,
             label='epoch:' + str(e_num) + 'th')
