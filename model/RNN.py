import torch
import torch.nn as nn


class PureRNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        num_layers=1,
        dropout=0,
        bidirectional=True,
        net_type="gru",
    ) -> None:
        super(PureRNNClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if net_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.reset_lstm_params()
        elif net_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif net_type.lower() == "vanilla":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                nonlinearity="relu",
                bidirectional=bidirectional,
            )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = x.view(batch_size, seq_len, self.input_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)

        return out
