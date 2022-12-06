import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score

def gen_xy(sequence):
    """
    Returns
    ---
    x: (num_samples, 5)
    y: (num_samples,) 1-d vec for labels
    """

    x, y = [], []
    for i in range(len(sequence) - 2):
        x.append(sequence[i : i + 2])
        y.append(sequence[i + 1])

    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x = x[:, :5]
    y = np.array(y)

    if len(y.shape) > 1:
        y = y[:, -1]

    return x, y

if __name__ == "__main__":
    data="../data/data_99771.pkl"
    
    sequence = pd.read_pickle(data)[
        ["from_user_id", "to_user_id", "label"]
    ].values

    x, y=gen_xy(sequence)

    train_size=0.7
    val_size=0.1
    cat_feat_index=[0, 1, 2, 3, 4]

    split1 = int(len(x) * train_size)
    split2 = int(len(sequence) * (train_size + val_size))

    x_train, y_train = x[:split1], y[:split1]
    x_val, y_val = x[split1:split2], y[split1:split2]
    x_test, y_test = x[split2:], y[split2:]

    print(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}")
    print(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}")
    print(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}")

    trainset=cbt.Pool(data=x_train, label=y_train, cat_features=cat_feat_index)
    valset=cbt.Pool(data=x_val, label=y_val, cat_features=cat_feat_index)
    testset=cbt.Pool(data=x_test, label=y_test, cat_features=cat_feat_index)

    cbt_params = {
        "iterations": 5000,
        "early_stopping_rounds": 100,
        "learning_rate": 0.05,
        "random_seed": 510,
        "loss_function": "CrossEntropy",
        "od_type": "Iter",
        "max_depth": 6,
        "verbose": 100,
    }

    model=cbt.CatBoostClassifier(**cbt_params)
    model.fit(trainset, eval_set=valset)

    y_train_pred=model.predict(x_train, verbose=True)
    y_val_pred=model.predict(x_val, verbose=True)
    y_pred=model.predict(x_test, verbose=True)

    print(accuracy_score(y_train, y_train_pred))
    print(accuracy_score(y_val, y_val_pred))
    print(accuracy_score(y_test, y_pred))
    