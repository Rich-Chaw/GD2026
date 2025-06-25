#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import graph


def GetSolution(stepRatio, model_file,data_test_path,save_dir):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    
    ## begin computing...
    sol_time_df = pd.DataFrame(np.arange(1*len(datasets)).reshape((1,len(datasets))),index=['time'], columns=datasets)

    for j in range(len(datasets)):
        print ('\nTesting dataset %s'%datasets[j])
        data_test = data_test_path + datasets[j] + '.txt'
        solution, time = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        sol_time_df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(datasets[j], time))
    sol_time_df.to_csv(save_dir + 'sol_time.csv' , encoding='utf-8', index=False)

    

def EvaluateSolution(stepRatio, model_file, strategyID, data_test_path,save_dir):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
   
    ## begin computing...
    score_df = pd.DataFrame(np.arange(2 * len(datasets)).reshape((2, len(datasets))), index=['solution', 'time'], columns=datasets)
    for i in range(len(datasets)):
        print('\nEvaluating dataset %s' % datasets[i])
        data_test = data_test_path + datasets[i] + '.txt'
        solution = save_dir + datasets[i] + '.txt'
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        ################################## modify to choose which strategy to evaluate
        score, MaxCCList = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=0.001)
        t2 = time.time()
        score_df.iloc[0, i] = score
        score_df.iloc[1, i] = t2 - t1
        result_file = save_dir + 'MaxCCList_Strategy_' + datasets[i] + '.txt'
        with open(result_file, 'w') as f_out:
            for j in range(len(MaxCCList)):
                f_out.write('%.8f\n' % MaxCCList[j])
        print('Data: %s, score:%.6f' % (datasets[i], score))
    score_df.to_csv(save_dir + 'solution_score.csv', encoding='utf-8', index=False)

def RandomRemoveEvaluate(STEPRATIO,REPEAT,MODEL_FILE_CKPT=None):

    # randomRatioList = [0.005,0.01,0.02,0.05,0.1,0.15,0.2]
    randomRatioList = [0.5]
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    # dqn = GraphDQN()
    cfd = os.path.dirname(__file__)
    ## data_set
    data_test_path = '%s/../../data/real/'%(cfd)
    # datasets = ['Crime','HI-II-14']
    ## model_file
    model_file_path = './FINDER_ND/models/barabasi_albert/'
    if not MODEL_FILE_CKPT:
        model_file_ckpt = dqn.findModel()
    model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + model_file_ckpt
    ## modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    ## save_dir : save sol
    save_dir = '%s/result/real'%(cfd)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ## save_dir_local : save random test result
    save_dir_local = save_dir + '/StepRatio_%.4f/evaluate_random' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
        
    ## begin computing...
    print("------------------ random evaluate ------------------")
    print('The best model is :%s'%(model_file))

    for j in range(len(datasets)):
        print ('\nTesting dataset %s'%datasets[j])
        data_test = data_test_path + datasets[j] + '.txt'
        df = pd.DataFrame(np.arange(REPEAT*len(randomRatioList)).reshape((REPEAT,len(randomRatioList))), columns=randomRatioList)
        for r_i,r in enumerate(randomRatioList):
            for t in range(REPEAT):
                score, MaxCCList,solution, time = dqn.EvaluateRealData_random(model_file, data_test, save_dir, r, stepRatio)
                print("score:",score)
                # print("solution",solution)
                df.iloc[t,r_i] = score
        print('Data:%s, remove ratio:%f, time:%.2f'%(datasets[j], r, time))
        df.to_csv(save_dir_local + '/%s.csv'%(datasets[j]), encoding='utf-8', index=False)

# datasets = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Facebook','Epinions','Youtube','Flickr']
datasets = ['Crime']
STEPRATIO = 0.01
STRTEGYID = 0               # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
dqn = GraphDQN()            # global ,why？？？？

def main():

    cfd = os.path.dirname(__file__)    

    model_file = None
    # model_file = 'nrange_10_30_iter_508800.ckpt'
    # model_file = 'nrange_30_50_iter_447000.ckpt'
    # model_file = 'nrange_50_70_iter_204900.ckpt'
    # model_file = 'nrange_70_90_iter_13500.ckpt'

    if model_file == None:  #if user do not specify the model_file
        model_file = dqn.findModel()
        model_file = '%s/models/%s/%s' % (cfd, dqn.g_type, model_file)
        print ('The best model is :%s'%(model_file))

    save_dir = '%s/result/real'%(cfd)
    save_dir_local = save_dir + '/%s/%d_%d_StepRatio_%.4f/' %(dqn.g_type,dqn.num_min,dqn.num_max,STEPRATIO)
    if not os.path.exists(save_dir_local):#make dir
            os.makedirs(save_dir_local)

    data_test_path = '%s/../../data/real/'%(cfd)


    # case 1
    # RandomRemoveEvaluate(0.01,100,model_file)
    
    # case 2
    GetSolution(STEPRATIO, model_file,data_test_path,save_dir = save_dir_local)
    EvaluateSolution(STEPRATIO, model_file, STRTEGYID,data_test_path,save_dir = save_dir_local)


    




if __name__=="__main__":
    main()
