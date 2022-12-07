import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import datetime

import warnings
warnings.filterwarnings("ignore")

def gen_xy(sequence):
    """
    Returns
    ---
    x: (num_samples, 2, num_features)
    y: (num_samples,) 1-d vec for labels
    """

    x, y = [], []
    for i in range(len(sequence) - 2):
        x.append(sequence[i : i + 2])
        y.append(sequence[i + 2 - 1])

    x = np.array(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = y[:, -1]

    return torch.FloatTensor(x), torch.LongTensor(y)


def get_dataloaders(sequence, train_size=0.7, val_size=0.1, batch_size=64):
    x, y = gen_xy(sequence)

    split1 = int(len(x) * train_size)
    split2 = int(len(sequence) * (train_size + val_size))

    x_train, y_train = x[:split1], y[:split1]
    x_val, y_val = x[split1:split2], y[split1:split2]
    x_test, y_test = x[split2:], y[split2:]

    print(f"Trainset:\tx-{x_train.size()}\ty-{y_train.size()}")
    print(f"Valset:  \tx-{x_val.size()}  \ty-{y_val.size()}")
    print(f"Testset:\tx-{x_test.size()}\ty-{y_test.size()}")

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True
    )

    return trainset_loader, valset_loader, testset_loader

class PL_MLP(pl.LightningModule):
    def __init__(
        self,
        num_users,
        embedding_dim=16,
        hidden_dim=32,
    ) -> None:
        super().__init__()

        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_dim = embedding_dim * 2 + 1

        self.user_embedding_layer = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim, padding_idx=-1
        )

        self.mlp = nn.Linear(self.input_dim, 2)
        
    def set_criterion(self, criterion):
        self.criterion=criterion
        
    def set_optimizer(self, optimizer):
        self.optimizer=optimizer

    def forward(self, x):
        # x: (batch_size, 2, 3(from, to, label))

        last_tx = x[:, 0, :]
        cur_tx = x[:, 1, :]

        last_from_user = last_tx[:, 0].long()
        last_label = last_tx[:, 2][:, None]

        cur_from_user = cur_tx[:, 0].long()

        last_from_embedding = self.user_embedding_layer(
            last_from_user
        )  # (batch_size, embedding_dim)

        cur_from_embedding = self.user_embedding_layer(cur_from_user)

        input = torch.concat(
            (
                last_from_embedding,
                cur_from_embedding,
                last_label,
            ),
            dim=1,
        )

        out = self.mlp(input)
        out = torch.softmax(out, dim=1)

        return out
    
    def training_step(self, batch, batch_idx):
        x, y=batch
        out=self.forward(x)
        loss=self.criterion(out, y)
        # self.log("Train loss", loss)
        return loss
    
    # def training_epoch_end(self, outputs) -> None:
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
    #     print(datetime.datetime.now(), "Epoch", self.current_epoch, avg_loss.cpu().numpy())
    
    def validation_step(self, batch, batch_idx):
        x, y=batch
        out=self.forward(x)
        loss=self.criterion(out, y)
        # self.log("Val loss", loss)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        num_users,
        embedding_dim=16,
        hidden_dim=64,
        num_layers=1,
        dropout=0,
        bidirectional=True,
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

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
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
    
    def set_criterion(self, criterion):
        self.criterion=criterion
        
    def set_optimizer(self, optimizer):
        self.optimizer=optimizer
        
    def training_step(self, batch, batch_idx):
        x, y=batch
        out=self.forward(x)
        loss=self.criterion(out, y)
        # self.log("Train loss", loss)
        return loss
    
    # def training_epoch_end(self, outputs) -> None:
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
    #     print(datetime.datetime.now(), "Epoch", self.current_epoch, avg_loss.cpu().numpy())
    
    def validation_step(self, batch, batch_idx):
        x, y=batch
        out=self.forward(x)
        loss=self.criterion(out, y)
        # self.log("Val loss", loss)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
if __name__ == "__main__":
    data = "../data/data_100000_distr.pkl"

    sequence = pd.read_pickle(data)[
        ["from_user_id", "to_user_id", "label"]
    ].values

    train_loader, val_loader, test_loader = get_dataloaders(
        sequence, batch_size=32, train_size=0.7, val_size=0.1
    )
    
    # model=PL_MLP(num_users=5000, embedding_dim=8, hidden_dim=16)
    model=RNNClassifier(num_users=5000, embedding_dim=8, hidden_dim=16)
    model.set_criterion(nn.CrossEntropyLoss())
    model.set_optimizer(torch.optim.Adam(model.parameters(), lr=1e-4))

    trainer=pl.Trainer(accelerator="cpu", max_epochs=10, enable_progress_bar=True, enable_checkpointing=False, profiler="simple")
    trainer.fit(model, train_loader, val_loader)
    