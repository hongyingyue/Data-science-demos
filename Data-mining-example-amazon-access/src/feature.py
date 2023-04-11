
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def transform_feature(data,feature_cols,target_cols=None,training=False):
    if training:
        assert target_cols is not None
        oh = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')

        x=oh.fit_transform(data.loc[:,feature_cols])
        with open("encoder", "wb") as f:
            pickle.dump(oh, f)

        y=data[target_cols].values
        print(x.shape,y.shape)
        return x,y

    with open('encoder','rb') as f:
        oh = pickle.load(f)
    x=oh.transform(data.loc[:,feature_cols])
    return x

