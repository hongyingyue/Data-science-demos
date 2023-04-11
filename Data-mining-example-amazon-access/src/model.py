
from data import load_data
from feature import transform_feature
import pandas as pd

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
seed=2020


def create_model():
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=seed,
        solver='liblinear',
        max_iter=1000,
    )
    return model


def validate_model(x,y,model,metrics):
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    re=[]

    for train_index, test_index in skf.split(x, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train,y_train)
        y_pred=model.predict_proba(x_test)[:,1]

        re.append(metrics(y_test,y_pred))

    print([round(i,3) for i in re])


def predict(model,x_test):
    predictions = model.predict_proba(x_test)[:, 1]
    print(predictions)
    return predictions


def main():
    train,test=load_data(data_dir='../data/amazon-employee-access-challenge')

    feature_cols=[i for i in train.columns if i not in ['ACTION','ROLE_TITLE']]
    target_cols='ACTION'
    x,y=transform_feature(train,feature_cols=feature_cols,target_cols=target_cols,training=True)

    model=create_model()
    result=validate_model(x,y,model,metrics=roc_auc_score)

    x_test = transform_feature(test, feature_cols=feature_cols,target_cols=None,training = False)
    predict(model,x_test)




if __name__=='__main__':
    main()
