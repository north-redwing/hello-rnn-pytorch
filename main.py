import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import get_data, get_pred_curve, draw_pred_curve
from net import TimeSeriesPredictor


def main():
    # params
    look_back = 5
    batch_size = 10
    max_epoch = 2000
    period = 400  # early stopping checking and drawing predicted curve period
    model = TimeSeriesPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # get data
    X_train, y_train, X_test, y_test, x, y = get_data(look_back=look_back)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    # learning curves
    learning_curve = []
    test_curve = []
    loss_prev = np.inf

    plt.figure(figsize=(9, 6))
    for epoch in range(1, max_epoch):
        # perm is for getting batch randomly
        perm = np.random.permutation(train_size)
        train_loss = 0

        # train loop
        model.train()
        for i in range(train_size // batch_size):
            optimizer.zero_grad()
            batch_x = X_train[perm[i * batch_size:(i + 1) * batch_size]]
            batch_y = y_train[perm[i * batch_size:(i + 1) * batch_size]]
            batch_x = batch_x.reshape(batch_size, -1, look_back).float()
            batch_z = model(batch_x).view(batch_size, )
            loss = criterion(batch_y, batch_z)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (train_size // batch_size)
        learning_curve.append(train_loss)

        # test
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for j in range(test_size):
                z = model(X_test[j].reshape(1, 1, look_back).float())
                z = z.view(1, )
                loss = criterion(y_test[j].view(1, ), z)
                test_loss += loss.item()
        test_curve.append(test_loss)

        # check early stopping
        # if test loss doesn't reduce by 1% of that at period epoch before,
        # # early stopping is done.
        if epoch % period == 0:
            print('epoch:{}, train loss:{}, test loss:{}'.format(
                epoch,
                train_loss,
                test_loss
            ))
            pred_curve = get_pred_curve(X_test, test_size, model, look_back)
            draw_pred_curve(train_size, look_back, x, y, pred_curve, epoch)
            if test_loss > loss_prev * 0.99:
                print('Stop learning')
                break
            else:
                loss_prev = test_loss

    # predicted curve
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title('prediction task on sin curve')
    plt.legend(loc='upper left', fontsize=10)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(learning_curve, color='blue', label='learning curve')
    plt.plot(test_curve, color='orange', label='test curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.legend(fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
