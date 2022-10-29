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
        # x: (batch_size, seq_len, 3)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x[:, :, 2]

        x = x.view(batch_size, seq_len, self.input_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)

        return out


class RNNClassifier(nn.Module):
    def __init__(
        self,
        num_users,
        embedding_dim=16,
        hidden_dim=64,
        num_layers=1,
        dropout=0,
        bidirectional=True,
        net_type="gru",
    ) -> None:
        super().__init__()

        self.num_users = num_users

        self.user_embedding_layer = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim, padding_idx=-1
        )
        
        # self.bn = nn.BatchNorm1d(num_features=embedding_dim)

        self.input_dim = embedding_dim * 2 + 1
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if net_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.reset_lstm_params()
        elif net_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif net_type.lower() == "vanilla":
            self.rnn = nn.RNN(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                nonlinearity="relu",
                bidirectional=bidirectional,
            )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_input_dim // 2, 2),
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 3(from, to, label))
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]

        from_user = x[:, :, 0].long()
        to_user = x[:, :, 1].long()
        label = x[:, :, 2][:, :, None]

        from_embedding = self.user_embedding_layer(
            from_user
        )  # (batch_size, seq_len, embedding_dim)
        # from_embedding = self.bn(from_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        
        to_embedding = self.user_embedding_layer(to_user)
        # to_embedding = self.bn(to_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        
        input = torch.concat(
            (from_embedding, to_embedding, label), dim=2
        )

        out, _ = self.rnn(input)
        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)

        return out
    
class RNNRegressor(nn.Module):
    def __init__(
        self,
        num_users,
        embedding_dim=16,
        hidden_dim=64,
        num_layers=1,
        dropout=0,
        bidirectional=True,
        net_type="gru",
    ) -> None:
        super().__init__()

        self.num_users = num_users

        self.user_embedding_layer = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim, padding_idx=-1
        )
        self.latency_embedding_layer = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU(inplace=True)
        )
        
        self.input_dim = embedding_dim * 3
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if net_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.reset_lstm_params()
        elif net_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif net_type.lower() == "vanilla":
            self.rnn = nn.RNN(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                nonlinearity="relu",
                bidirectional=bidirectional,
            )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_input_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 3(from, to, label))
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]

        from_user = x[:, :, 0].long()
        to_user = x[:, :, 1].long()
        latency = x[:, :, 2][:, :, None]

        from_embedding = self.user_embedding_layer(
            from_user
        )  # (batch_size, seq_len, embedding_dim)
        # from_embedding = self.bn(from_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        
        to_embedding = self.user_embedding_layer(to_user)
        # to_embedding = self.bn(to_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        
        latency_embedding = self.latency_embedding_layer(latency)
        
        input = torch.concat(
            (from_embedding, to_embedding, latency_embedding), dim=2
        )

        out, _ = self.rnn(input)
        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)

        return out
