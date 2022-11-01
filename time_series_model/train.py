import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import accuracy
from RNN import PureRNNClassifier, RNNClassifier


def gen_xy(sequence, window):
    """
    Returns
    ---
    x: (num_samples, seq_len)
    y: (num_samples,) 1-d vec for labels
    """

    x, y = [], []
    for i in range(len(sequence) - window):
        x.append(sequence[i : i + window])
        y.append(sequence[i + window])

    x = np.array(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = y[:, -1]

    return torch.FloatTensor(x), torch.LongTensor(y)


def get_dataloaders(sequence, window, train_size=0.7, val_size=0.1, batch_size=256):
    split1 = int(len(sequence) * train_size)
    split2 = int(len(sequence) * (train_size + val_size))

    train_data = sequence[:split1]
    val_data = sequence[split1:split2]
    test_data = sequence[split2:]

    x_train, y_train = gen_xy(train_data, window)
    x_val, y_val = gen_xy(val_data, window)
    x_test, y_test = gen_xy(test_data, window)

    print(f"Trainset:\tx-{x_train.size()}\ty-{y_train.size()}")
    print(f"Valset:  \tx-{x_val.size()}  \ty-{y_val.size()}")
    print(f"Testset:\tx-{x_test.size()}\ty-{y_test.size()}")

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train_one_epoch(model, trainset_loader, optimizer, criterion):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    max_epochs=100,
    early_stop=10,
    verbose=1,
    plot=False,
    log="train.log",
):
    if log:
        log = open(log, "a")
        log.seek(0)
        log.truncate()

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainset_loader, optimizer, criterion
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss, val_acc = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if (epoch + 1) % verbose == 0:
            print(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Train acc = %.5f " % train_acc,
                "Val Loss = %.5f" % val_loss,
                "Val acc = %.5f " % val_acc,
            )

            if log:
                print(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    "\tTrain Loss = %.5f" % train_loss,
                    "Train acc = %.5f " % train_acc,
                    "Val Loss = %.5f" % val_loss,
                    "Val acc = %.5f " % val_acc,
                    file=log,
                )
                log.flush()

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                print(f"Early stopping at epoch: {epoch+1}")
                print(f"Best at epoch {best_epoch+1}:")
                print(
                    "Train Loss = %.5f" % train_loss_list[best_epoch],
                    "Train acc = %.5f " % train_acc_list[best_epoch],
                )
                print(
                    "Val Loss = %.5f" % val_loss_list[best_epoch],
                    "Val acc = %.5f " % val_acc_list[best_epoch],
                )

                if log:
                    print(f"Early stopping at epoch: {epoch+1}", file=log)
                    print(f"Best at epoch {best_epoch+1}:", file=log)
                    print(
                        "Train Loss = %.5f" % train_loss_list[best_epoch],
                        "Train acc = %.5f " % train_acc_list[best_epoch],
                        file=log,
                    )
                    print(
                        "Val Loss = %.5f" % val_loss_list[best_epoch],
                        "Val acc = %.5f " % val_acc_list[best_epoch],
                        file=log,
                    )
                    log.flush()
                break

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(range(0, epoch + 1), train_acc_list, "-", label="Train Acc")
        plt.plot(range(0, epoch + 1), val_acc_list, "-", label="Val Acc")
        plt.title("Epoch-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    if log:
        log.close()

    torch.save(best_state_dict, "./saved/best_state_dict.pkl")
    model.load_state_dict(best_state_dict)
    return model


if __name__ == "__main__":
    net_type = "gru"
    window = 5
    num_users = 2000
    embedding_dim = 8
    hidden_dim = 32
    batch_size = 64
    num_layers = 1
    dropout = 0
    lr = 1e-5
    bi = False

    max_epochs = 200
    log_file = "train.log"

    if torch.backends.mps.is_available():
        # DEVICE = torch.device("mps")
        DEVICE = torch.device("cpu")
    elif torch.cuda.is_available():
        GPU_ID = 0
        DEVICE = torch.device(f"cuda:{GPU_ID}")
    else:
        DEVICE = torch.device("cpu")

    for net_type in ("gru", "lstm", "vanilla"):
        for num in (9969, 49986, 99771):
            log_file = f"{net_type}_{num}.log"
            sequence = pd.read_pickle(f"../data/data_{num}.pkl")[
                ["from_user_id", "to_user_id", "label"]
            ].values

            train_loader, val_loader, test_loader = get_dataloaders(
                sequence, window, batch_size=batch_size
            )

            model = RNNClassifier(
                num_users=num_users,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                net_type=net_type,
                num_layers=num_layers,
                bidirectional=bi,
                dropout=dropout,
            ).to(DEVICE)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                max_epochs=max_epochs,
                early_stop=10,
                verbose=1,
                plot=False,
                log=log_file,
            )

            test_loss, test_acc = eval_model(model, test_loader, criterion)
            print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc)
            with open(log_file, "a") as f:
                print(
                    "Test Loss = %.5f" % test_loss,
                    "Test acc = %.5f " % test_acc,
                    file=f,
                )

    # num = 99771
    # sequence = pd.read_pickle(f"../data/data_{num}.pkl")[
    #     ["from_user_id", "to_user_id", "label"]
    # ].values

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     sequence, window, batch_size=batch_size
    # )

    # model = RNNClassifier(
    #     num_users=num_users,
    #     embedding_dim=embedding_dim,
    #     hidden_dim=hidden_dim,
    #     net_type=net_type,
    #     num_layers=num_layers,
    #     bidirectional=bi,
    #     dropout=dropout,
    # ).to(DEVICE)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # model = train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     criterion,
    #     max_epochs=max_epochs,
    #     early_stop=10,
    #     verbose=1,
    #     plot=False,
    #     log=log_file,
    # )

    # test_loss, test_acc = eval_model(model, test_loader, criterion)
    # print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc)
    # with open(log_file, "a") as f:
    #     print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc, file=f)
