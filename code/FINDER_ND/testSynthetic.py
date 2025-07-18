#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
from tqdm import tqdm


def main():
    dqn = GraphDQN()

    cfd = os.path.dirname(__file__)
    data_test_path = '%s/../../data/synthetic/uniform_cost/'%(cfd)
    data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    # data_test_name = ['30-50']

    
    result_path = '%s/result/synthetic'%(cfd)
    
    if not os.path.exists('%s/result'%cfd):
        os.mkdir('%s/result'%cfd)
    if not os.path.exists('%s/result/synthetic'%(cfd)):
        os.mkdir('%s/result/synthetic'%(cfd))
        
    with open('%s/result.txt'%result_path, 'w',encoding="utf-8") as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            # if model_file is not set, dqn.Evaluate will check the model dir to find best
            # score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, model_file)
            score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test)
            fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
            fout.flush()
            print('\ndata_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    main()
