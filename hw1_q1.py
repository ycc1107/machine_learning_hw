#!/opt/anaconda/bin/python
from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import multiprocessing as mp
import pandas as pd

def calc_func(data_feature, test_feature, data_labels, test_labels):
  arg2 = -2 * np.dot(data_feature, np.transpose(test_feature))
  arg1 = (test_feature * test_feature).sum(axis=1)[np.newaxis,:]
  arg3 = (data_feature * data_feature).sum(axis=1)[:,np.newaxis]
  dist = arg1 + arg2 + arg3

  res_idx = [np.argmin(line) for line in dist]
  res = data_labels - test_labels[res_idx]

  return res[np.where(res == 0)].shape[0]
 
def _worker(shared_dict, raw_data, training_size, labels, num_iter=10):
  if training_size > len(raw_data['data']):
    raise Exception('Training size can\'t larger than data size')

  train_data, train_label, test_data, test_label = labels
  for _ in xrange(num_iter):
    total_size = len(raw_data['data'])
    sel_data = random.sample(xrange(total_size), training_size)
    data_feature = raw_data[train_data][sel_data].astype('float')
    data_labels = raw_data[train_label][sel_data].astype('float')
    test_feature = raw_data[test_data].astype('float')
    test_labels = raw_data[test_label].astype('float')

    res = calc_func(data_feature, test_feature, data_labels, test_labels)    

    shared_dict[training_size] = shared_dict.get(training_size, []) + [res]

def calc_data(raw_data, labels, training_set=None, num_iter=10):
    if not training_set:
      training_set = (1000, 2000, 4000, 8000)
    
    shared_dict = mp.Manager().dict()
    jobs = []

    for num in training_set:
      process = mp.Process(target=_worker, args=(shared_dict, raw_data, num, labels, num_iter))
      process.start()
      jobs.append(process)

    [p.join() for p in jobs]

    res = []
    for idx in training_set:
      value = shared_dict[idx]
      res.append([idx, np.mean(value)/idx, np.std(value)])

    res_df = pd.DataFrame(res, columns=['Iter_Num', 'Mean', 'STD'])
    return res_df
        
def plot_result(data):
    x_axis = range(len(data))
    y_axis = np.mean(data, axis=1)
    err = np.std(data, axis=1)

    plt.errorbar(x_axis, y_axis, yerr=err, fmt='o')
    plt.show()

def run():
    ocr = loadmat('ocr.mat')
    print 'Using test data:'
    labels = ['data', 'labels', 'testdata', 'testlabels']
    print calc_data(ocr, labels)

    print 'Using test data:'
    labels = ['data', 'labels', 'data', 'labels']
    print calc_data(ocr, labels)

if  __name__ == '__main__':
    run()
