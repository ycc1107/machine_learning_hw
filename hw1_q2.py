from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import combinations
import pandas as pd

mapping = {'1'  : 'fixed acidity',
           '2'  : 'volatile acidity',
           '3'  : 'citric acid',
           '4'  : 'residual sugar',
           '5'  : 'chlorides',
           '6'  : 'free sulfur dioxide',
           '7'  : 'total sulfur dioxide',
           '8'  : 'density',
           '9'  : 'pH',
           '10' : 'sulphates',
           '11' : 'alcohol',}

def banch_mark(wine):
    regr = LinearRegression()
    train_data_x = wine['data']
    train_data_y = wine['labels']
    regr.fit(train_data_x, train_data_y)
    pred_y = regr.predict(wine['testdata'])

    return mean_squared_error(pred_y, wine['testlabels'])

def customized_pre_model(wine):
    regr = LinearRegression()
    data = wine['data']
    real_y = wine['testlabels']
    test_data = wine['testdata']
    train_data_y = wine['labels']
    
    min_score = 10e10
    for a, b, c  in combinations(range(len(data[0])), r=3):
        temp = np.column_stack((data[:, a], data[:, b], data[:, c]))
        test_data_tmp = np.column_stack((test_data[:, a], test_data[:, b], test_data[:, c]))
        regr.fit(temp, train_data_y)
        pred_y = regr.predict(test_data_tmp)
        loss = mean_squared_error(pred_y ,real_y)
        score = regr.score(test_data_tmp, real_y)
        if loss < min_score:
            min_score = loss
            best_ = loss
            idx_ = '{}.{}.{}'.format(a, b, c)
            print idx_ ,score, loss
            coef = regr.coef_

    res_df = pd.DataFrame([coef[0]], columns=[mapping[i] for i in idx_.split('.')])
    res_df.index = ['coefficient']

    return res_df


def run():
    wine = loadmat('wine.mat')
    banch_mark_res = banch_mark(wine)
    print 'banch mark loss: {}'.format(banch_mark_res)
    
    res_df = customized_pre_model(wine)
    print res_df

if __name__ == '__main__':
    run()

