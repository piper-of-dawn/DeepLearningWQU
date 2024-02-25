from sklearn.preprocessing import LabelEncoder
from collections import namedtuple
def encode_labels(data, columns):
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data


def train_test_split(data, test_size, y_name):
    train_test = namedtuple('train_test', ['x_train', 'x_test', 'y_train', 'y_test'])
    split_row = len(data) - int(test_size * len(data))
    train_data = data.iloc[:split_row]
    test_data = data.iloc[split_row:]
    return train_test(x_train=train_data.drop(y_name, axis=1).to_numpy(), x_test=test_data.drop(y_name, axis=1).to_numpy(), y_train=train_data[y_name].to_numpy(), y_test=test_data[y_name].to_numpy())