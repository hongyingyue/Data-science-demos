
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def load_data(data_dir,plot=False):
    train = pd.read_csv(data_dir+'/train.csv')
    test = pd.read_csv(data_dir+'/test.csv')
    print(train.values.shape,test.values.shape)
    print(train.head(5))
    print(np.unique(train.loc[:,['ACTION']]))

    if plot:
        dis=train.groupby(['ACTION']).size().reset_index(name='size')

        plt.bar(dis['ACTION'].astype('int'),dis['size'],color='lightblue')
        for a, b in zip(dis['ACTION'], dis['size']):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.text(a, b/3, '%.0f'%a, ha='center',va='bottom',fontsize=15,color='blue')
        plt.show()
    return train,test



if __name__=='__main__':
    data_dir='../data/amazon-employee-access-challenge'
    load_data(data_dir)