#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import heapq
import scipy.linalg as linalg
import os
import pandas as pd
# from gurobipy import *

# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 1000000
cdef double LEARNING_RATE = 0.0001   #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30
cdef int NUM_MAX = 50
cdef int REG_HIDDEN = 32
cdef int M = 4  # how many edges selected each time for BA model
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.01  # 权重初始化的方差
cdef int n_valid = 200
cdef int aux_dim = 4
cdef int num_env = 1
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 2
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage



class GraphDQN:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'small-world'#erdos_renyi, powerlaw, small-world
        self.training_type = 'degree'   #'random'
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False

        self.IsPrioritizedSampling = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        # self.covered=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)
        tf.compat.v1.disable_eager_execution()
        # [batch_size, node_cnt]
        self.action_select = tf.compat.v1.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.compat.v1.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.compat.v1.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf.compat.v1.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.compat.v1.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE,1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf.compat.v1.placeholder(tf.float32, name="aux_input")
        # [node_cnt, 2]
        self.node_input = tf.compat.v1.placeholder(tf.float32, name="node_input")

        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')


        # init Q network
        self.loss,self.trainStep,self.q_pred, self.q_on_all,self.Q_param_list = self.BuildNet() #[loss,trainStep,q_pred, q_on_all, ...]
        #init Target Q Network
        self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT = self.BuildNet()
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)]


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf.compat.v1.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config = config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.compat.v1.global_variables_initializer())

#################################################New code for graphDQN#####################################
    def BuildNet(self):
        # [2, embed_dim]
        w_n2l = tf.Variable(tf.compat.v1.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)
        # [embed_dim, embed_dim]
        p_node_conv = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 1:    #'graphsage'
            # [embed_dim, embed_dim]
            p_node_conv2 = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim]
            p_node_conv3 = tf.Variable(tf.compat.v1.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            #[2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.compat.v1.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.compat.v1.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim, 1]
            h2_weight = tf.Variable(tf.compat.v1.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            #[reg_hidden2 + aux_dim, 1]
            last_w = h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.compat.v1.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim, 1]
        cross_product = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)

        # #[node_cnt, 2]
        # nodes_size = tf.shape(self.n2nsum_param)[0]
        # node_input = tf.ones((nodes_size,2))

        y_nodes_size = tf.shape(self.subgsum_param)[0]
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # no sparse
        input_message = tf.matmul(tf.cast(self.node_input,tf.float32), w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        input_potential_layer = tf.nn.relu(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse
        y_input_potential_layer = tf.nn.relu(y_input_message)

        #input_potential_layer = input_message
        cdef int lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)


        while lv < max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            n2npool = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)
            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)

            if embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message)
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        # y_potential = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
        #[[batch_size, embed_dim], [batch_size, embed_dim]] = [batch_size, 2*embed_dim], dense
        # embed_s_a = tf.concat([action_embed,y_potential],1)
        # # [batch_size, embed_dim, embed_dim]
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
        # [batch_size, embed_dim]
        Shape = tf.shape(action_embed)
        # [batch_size, embed_dim], first transform
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)

        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)
            #[batch_size, reg_hidden1] * [reg_hidden1, reg_hidden2] = [batch_size, reg_hidden2]
            # last_output_hidden = tf.matmul(last_output1, h2_weight)
            # last_output = tf.nn.relu(last_output_hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        ## first order reconstruction loss
        loss_recons = 2 * tf.compat.v1.trace(tf.matmul(tf.transpose(cur_message_layer), tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer)))
        edge_num = tf.compat.v1.sparse_reduce_sum(self.n2nsum_param)
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)

        loss = loss_rl + Alpha * loss_recons
        #
        # loss = loss_rl

        # self.lossRecons = loss_recons
        # self.lossRL = loss_rl

        trainStep = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(loss)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)
        #[[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
        # embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)
        # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))
        # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # [batch_size, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]
            # last_output_hidden = tf.matmul(last_output1, h2_weight)
            # last_output = tf.nn.relu(last_output_hidden)

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss,trainStep,q_pred,q_on_all,tf.compat.v1.trainable_variables()


    def gen_graph(self, num_min, num_max):
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)

        if self.training_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
        else:
            degree = nx.degree_centrality(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]
        nx.set_node_attributes(g, weights, 'weight')
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        cdef int i
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    #pass
    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        cdef double result_degree = 0.0
        cdef double result_betweeness = 0.0

        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            g_degree = g.copy()
            g_betweenness = g.copy()
            result_degree += self.HXA(g_degree,'HDA')
            result_betweeness += self.HXA(g_betweenness, 'HBA')
            self.InsertGraph(g, is_test=True)
        print ('Validation of DegreeGreedy: %.16f'%(result_degree / n_valid))
        print ('Validation of BetweenessGreedy: %.16f'%(result_betweeness / n_valid))


    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        while n < num_seq:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                    g_sample= TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)
    #pass
    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat


    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list

    def Predict(self,g_list,covered,isSnapSnot):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result
    #pass
    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        cdef int i
        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input : np.array(self.inputs['aux_input']),
                                        self.ISWeights : np.mat(ISWeights).T,
                                        self.target : self.inputs['target']})
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        # print ('loss')
        # print (loss / len(g_list))
        return loss / len(g_list)


    def fit(self,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.loss,self.trainStep],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input: np.array(self.inputs['aux_input']),
                                        self.target : self.inputs['target']})
            loss += result[0]*bsize
            # print ('Reconstruction loss')
            # print (result[2])
            # print ('RL loss')
            # print (result[3])

        return loss / len(g_list)
    #pass

    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)
        cdef int i, iter, idx
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        cdef double eps_start = 1.0
        cdef double eps_end = 0.05
        cdef double eps_step = 10000.0
        cdef int loss = 0
        cdef double frac, start, end

        save_dir = './models/Model_%s'%(self.g_type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            start = time.clock()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            ###########-----------------------normal training data setup(end) -----------------##############################

            ###########-----------------------stratified training data setup(start) -----------------##############################
            # if iter <= 200000:
            #     if iter and iter % 5000 == 0:
            #         self.gen_new_graphs(15, 20)
            # elif iter > 200000 and iter <= 400000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(30, 50)
            # elif iter > 400000 and iter <= 600000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(50, 100)
            # elif iter > 600000 and iter <= 800000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(100, 200)
            # elif iter > 800000 and iter <= 1000000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(400, 500)
            ###########-----------------------stratified training data setup(end) -----------------##############################
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                # n_valid = 1
                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush()
                print('iter', iter, 'eps', eps, 'average size of vc: ', frac / n_valid)
                print ('testing 100 graphs time: %.8fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.8fs'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            self.Fit()
        f_out.close()


    def findModel(self):
        VCFile = './models/ModelVC_%d_%d.csv'%(NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        best_model = './models/nrange_%d_%d_iter_%d.ckpt' % (NUM_MIN, NUM_MAX, best_model_iter)
        return best_model

    def Evaluate1(self, g, save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        result_file = '%s/test.csv' % (save_dir)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(0)
            t2 = time.time()
            for i in range(len(sol)):
                f_out.write(' %d\n' % sol[i])
            frac += val
            frac_time += (t2 - t1)
        print ('average size of vc: ', frac)
        print('average time: ', frac_time)


    def Evaluate(self, data_test, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef int i
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        self.ClearTestGraphs()
        # print ('\nvc mean: ', np.mean(result_list))
        # print ('vc std: ', np.std(result_list))
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std

    def EvaluateReinsert(self, data_test,save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        cdef int i
        f = open(data_test, 'rb')
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = cp.load(f)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSolReinsert(i)
            t2 = time.time()
            frac += val
            frac_time += (t2 - t1)
        self.ClearTestGraphs()
        print ('average size of vc: ', frac / n_test)
        print('average time: ', frac_time / n_test)
        return  frac / n_test, frac_time / n_test

    def CleanRealData(self, data_test):
        G = nx.read_weighted_edgelist(data_test)
        cc = sorted(nx.connected_components(G), key = len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        for node in G.nodes():
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges_iter(data=True):
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            # w = edge[2]['weight']
            # g.add_edge(src_idx,dst_idx,weight=w)
            g.add_edge(src_idx,dst_idx)
        nx.write_edgelist(g, './Real_data/real1.txt')


    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio=0.0025):  #测试真实数据
        cdef double solution_time = 0.0
        test_name = data_test.split('/')[-1].replace('.gml','.txt')
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file = '%s/%s' %(save_dir_local, test_name)
        # g = nx.read_edgelist(data_test)
        g = nx.read_gml(data_test)
        # g = self.Real2networkx(g_temp)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            if stepRatio > 0:
                step = int(stepRatio*nx.number_of_nodes(g)) #step size
            else:
                step = 1
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            solution = self.GetSolution(0,step)
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])
        self.ClearTestGraphs()
        return solution, solution_time

    def EvaRealForModelSelect(self, model_file, data_test, stepRatio=0.0025):  #测试真实数据
        g = nx.read_gml(data_test)
        g_inner = self.GenNetwork(g)
        if stepRatio > 0:
            step = int(stepRatio*nx.number_of_nodes(g)) #step size
        else:
            step = 1
        self.InsertGraph(g, is_test=True)
        sol = self.GetSolution(0,step)
        self.ClearTestGraphs()
        nodes = list(range(nx.number_of_nodes(g)))
        sol_left = list(set(nodes)^set(sol))
        solution = sol + sol_left
        Robustness = self.utils.getRobustness(g_inner, solution)
        return Robustness

    def GetSolution(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        return sol


    def EvaluateSol(self, data_test, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution, strategyID:0,count;2:rank;3:multipy
        sys.stdout.flush()
        # g = nx.read_weighted_edgelist(data_test)
        g = nx.read_gml(data_test)
        g_inner = self.GenNetwork(g)
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, reInsertStep)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution))
        Robustness = self.utils.getRobustness(g_inner, solution)
        MaxCCList = self.utils.MaxWccSzList
        return Robustness, MaxCCList, solution


    def Test(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        cdef int i
        sol = []
        while (not self.test_env.isTerminal()):
            # cost += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness


    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        # end = time.time()
        # print ('solution obtained time is:%.8f'%(end-start))
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        # print ('Robustness is:%.8f'%(Robustness))
        return Robustness, sol

    def GetSolReinsert(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        nodes = list(range(g_list[0].num_nodes))
        sol_left = list(set(nodes)^set(sol))
        sol_reinsert = self.utils.reInsert(g_list[0], sol, sol_left, 1, 1)
        solution = sol_reinsert + sol_left
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol

    def SaveModel(self,model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def loadGraphList(self, graphFileName):
        with open(graphFileName, 'rb') as f:
            graphList = cp.load(f)
        print("load graph file success!")
        return graphList


    def GenNetwork(self, g):    #networkx2four
        nodes = g.nodes()
        edges = g.edges()
        weights = []
        for i in range(len(nodes)):
            weights.append(g.node[str(i)]['weight'])
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
            W = np.array(weights)
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
        return graph.py_Graph(len(nodes), len(edges), A, B, W)


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

    def Isterminal(self, graph):
        if len(nx.edges(graph)) == 0:
            return True
        else:
            return False

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', 'HCA'
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(node)
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        return Robustness

    def Real2networkx(self, G):
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        G_copy = G.copy()
        nodes = G_copy.nodes()
        weights = {}
        for node in nodes:
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            weights[numrealnodes] = G_copy.node[node]['weight']
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges():
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            g.add_edge(src_idx, dst_idx)
        nx.set_node_attributes(g, 'weight', weights)
        return g

    def Evaluate4Vis(self, g, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        print ('testing')
        sys.stdout.flush()
        self.InsertGraph(g, is_test=True)
        sol = self.GetSol4Vis(0)
        print (sol)
        fout_sol = open('./Visualize/PKL/Sol.pkl','ab')
        cp.dump(sol, fout_sol)
        fout_sol.close()


    def GetSol4Vis(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        cdef int iter = 0
        while (not self.test_env.isTerminal()):
            fout_q = open('./Visualize/PKL/Q_%d.pkl'%iter,'ab')
            fout_embed = open('./Visualize/PKL/Embed_%d.pkl'%iter,'ab')
            list_pred, list_embed = self.Predict4Vis(g_list, [self.test_env.action_list])
            cp.dump(list_pred[0], fout_q)
            cp.dump(list_embed[0], fout_embed)
            fout_q.close()
            fout_embed.close()
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
            iter += 1
        return sol


    def Predict4Vis(self,g_list,covered):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            q_all, node_embed = self.session.run([self.q_on_all, self.node_embedding], feed_dict={
                self.rep_global: self.inputs['rep_global'],
                self.n2nsum_param: self.inputs['n2nsum_param'],
                self.subgsum_param: self.inputs['subgsum_param'],
                self.node_input: self.inputs['node_input'],
                self.aux_input: np.array(self.inputs['aux_input'])})

            raw_output = q_all
            raw_embed = node_embed
            # print("#####@!@@@")
            # print(np.shape(node_embed[0]))
            pos = 0
            pred = []
            embed = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                cur_embed = []
                for m in range(len(idx_map)):
                    cur_embed.append([])
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                        cur_embed[k] = []
                    else:
                        cur_pred[k] = raw_output[pos]
                        cur_embed[k] = raw_embed[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                    cur_embed[k] = []
                pred.append(cur_pred)
                embed.append(cur_embed)
            assert (pos == len(raw_output))
        return pred,embed
