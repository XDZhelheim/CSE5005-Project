import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
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

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.BatchNorm1d(num_features=self.hidden_dim),
        #     nn.Dropout(0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hidden_dim, 2),
        # )
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.input_dim, 2),
        #     nn.BatchNorm1d(num_features=2),
        #     nn.Dropout(0.2, inplace=True),
        # ) # 0.8
        
        self.mlp = nn.Linear(self.input_dim, 2)

    def forward(self, x):
        # x: (batch_size, 2, 3(from, to, label))

        last_tx = x[:, 0, :]
        cur_tx = x[:, 1, :]

        last_from_user = last_tx[:, 0].long()
        # last_to_user = last_tx[:, 1].long()
        last_label = last_tx[:, 2][:, None]

        cur_from_user = cur_tx[:, 0].long()
        # cur_to_user = cur_tx[:, 1].long()

        last_from_embedding = self.user_embedding_layer(
            last_from_user
        )  # (batch_size, embedding_dim)
        # last_to_embedding = self.user_embedding_layer(last_to_user)

        cur_from_embedding = self.user_embedding_layer(cur_from_user)
        # cur_to_embedding = self.user_embedding_layer(cur_to_user)

        input = torch.concat(
            (
                last_from_embedding,
                # last_to_embedding,
                cur_from_embedding,
                # cur_to_embedding,
                last_label,
            ),
            dim=1,
        )

        out = self.mlp(input)
        out = torch.softmax(out, dim=1)

        return out
