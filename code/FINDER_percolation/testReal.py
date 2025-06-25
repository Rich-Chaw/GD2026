#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN

def main():
    dqn = GraphDQN()
    # data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = ['Crime', 'HI-II-14']
    model_file = './FINDER_percolation/models/nrange_30_50_iter_753600.ckpt'
    
    STEPRATIO = 0.01
    
    if not os.path.exists('../results/FINDER_percolation'):
        os.mkdir('../results/FINDER_percolation')
    if not os.path.exists('../results/FINDER_percolation/real'):
        os.mkdir('../results/FINDER_percolation/real')
    data_save_path = '../results/FINDER_percolation/real/StepRatio_%.4f/'%STEPRATIO
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)

    dqn.GetCoreDQNSolution(model_file, data_test_name, data_save_path, STEPRATIO)



if __name__=="__main__":
    main()
