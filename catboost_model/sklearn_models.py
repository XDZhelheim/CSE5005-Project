import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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

def train(model, x_train, y_train, x_val, y_val, x_test, y_test):
    print(type(model))
    
    model.fit(x_train, y_train)

    y_train_pred=model.predict(x_train)
    y_val_pred=model.predict(x_val)
    y_pred=model.predict(x_test)

    print(accuracy_score(y_train, y_train_pred))
    print(accuracy_score(y_val, y_val_pred))
    print(accuracy_score(y_test, y_pred))

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
    
    model_list=[
        KNeighborsClassifier(n_neighbors=5),
        tree.DecisionTreeClassifier(criterion="gini", max_depth=3),
        tree.ExtraTreeClassifier(criterion="gini", splitter="best", max_depth=5),
        RandomForestClassifier(n_estimators=50, max_depth=5),
        SVC(),
    ]
    
    for model in model_list:
        train(model, x_train, y_train, x_val, y_val, x_test, y_test)
    