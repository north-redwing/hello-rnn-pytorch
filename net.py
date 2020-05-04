import torch.nn as nn


class TimeSeriesPredictor(nn.Module):
    def __init__(self):
        super(TimeSeriesPredictor, self).__init__()

        self.RNN = nn.LSTM(
            input_size=5,
            hidden_size=2,
            batch_first=True
        )
        self.fc = nn.Linear(2, 1)

    def forward(self, xs):
        hs, (hT, _) = self.RNN(xs)
        ys = self.fc(hs)

        return ys
