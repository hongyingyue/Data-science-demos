
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def train():
    data = pd.read_csv('data/Train_use.csv')
    data = data.iloc[:, 1:]

    category_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    cat_data = onehot_encoder.fit_transform(data.loc[:, category_cols]).toarray()

    num_data = data.loc[:, [i for i in data.columns if i not in category_cols + ['price_log']]].values

    x_train = np.concatenate([num_data, cat_data], axis=1)
    y_train = data.loc[:, ['price_log']].values

    print(cat_data.shape, num_data.shape)
    print(x_train.shape, y_train.shape)

    model = LinearRegression()

    reg = model.fit(x_train, y_train)
    print(reg.score(x_train, y_train))


if __name__ == '__main__':
    train()
