from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def min_idx(seq):
    it_obj = iter(seq)
    res = 0
    try: 
        min_num = it_obj.next()
    except StopIteration: 
        raise ValueError("the sequence is empty")
        
    for i,item in enumerate(it_obj,start=1):
        if item > min_num:
            min_num = item
            res = i
    return res

def calc_func(data_feature, test_feature, data_labels, test_labels):
    arg2 = -2 * np.dot(data_feature, np.transpose(test_feature))
    arg1 = (test_feature * test_feature).sum(axis=1)[np.newaxis,:]
    arg3 = (data_feature * data_feature).sum(axis=1)[:,np.newaxis]
    
    dist = arg1 + arg2 + arg3

    res_idx = [min_idx(line) for line in dist]
    res = data_labels - test_labels[res_idx]

    return res

def calc_data(raw_data):
    nums = (1000, 2000, 4000, 8000)
    res_dist = {}
    for _ in xrange(10):
        for num in nums:
            total_size = len(raw_data['data'])
            sel_data = random.sample(xrange(total_size), num)
            data_feature = raw_data['data'][sel_data].astype('float')
            data_labels = raw_data['labels'][sel_data].astype('float')
            test_feature =  raw_data['testdata'].astype('float')
            test_labels =  raw_data['testlabels'].astype('float')

            res = calc_func(data_feature, test_feature, data_labels, test_labels)    

            res_dist[num] = res_dist.get(num, []) + [res[np.where(res == 0)].shape[0]]

    for k, v in res_dist.iteritems():
        print 'samples size: {} avg_mean: {} all_iter_data:{}'.format(k, sum(v)/len(v), v)
    
    return res_dist.values()
        
def plot_result(data):
    x_axis = range(len(data))
    y_axis = np.mean(data, axis=1)
    err = np.std(data, axis=1)

    plt.errorbar(x_axis, y_axis, yerr=err, fmt='o')
    plt.show()

def run():
    ocr = loadmat('ocr.mat')
    calc_data(ocr)

if  __name__ == '__main__':
    run()
