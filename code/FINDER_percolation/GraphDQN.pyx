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
import mvc_env
import TreeBreaker
import utils

# Hyper Parameters:
cdef int GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int max_bp_iter = 5
cdef int MAX_ITERATION = 1000000
cdef double LEARNING_RATE = 0.0001  #dai
cdef int MEMORY_SIZE = 500000
cdef int N_STEP = 5
cdef int NUM_MIN = 50
cdef int NUM_MAX = 100
cdef int REG_HIDDEN = 64
cdef int M = 4  # how many edges selected each time for BA model
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.01  # 权重初始化的方差
cdef int n_valid = 100
cdef int aux_dim = 3
cdef int num_env = 1
cdef double inf = 2147483647/2



class GraphDQN:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.isDualNetwork = False
        self.g_type = 'barabasi_albert'
        # self.TrainSet = GSet()
        self.TrainSet = graph.py_GSet()
        # self.TestSet = GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.m_y = 0.0
        self.IsDoubleDQN = False
        self.utils = utils.py_Utils()

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.g_nx_list = []
        # self.covered=[]
        self.pred=[]
        # self.nStepReplayMem=NStepReplayMem(MEMORY_SIZE)
        self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        cdef int i
        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(self.GenNetwork(nx.null_graph()))
            self.g_nx_list.append(nx.null_graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)
        tf.compat.v1.disable_eager_execution()
        # [batch_size, node_cnt]
        self.action_select = tf.compat.v1.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.compat.v1.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.compat.v1.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.compat.v1.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.label = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE,1], name="label")
        # [batch_size, aux_dim]
        self.aux_input = tf.compat.v1.placeholder(tf.float32, name="aux_input")


        # init Q network
        self.loss,self.trainStep,self.q_pred, self.q_on_all,self.w_n2l, self.p_node_conv, self.h1_weight, self.h2_weight = self.BuildNet()
        #init Target Q Network
        self.lossT, self.trainStepT,self.q_predT, self.q_on_allT, self.w_n2lT, self.p_node_convT, self.h1_weightT, self.h2_weightT= self.BuildNet()
        #takesnapsnot
        self.copyTargetQNetworkOperation = [self.w_n2lT.assign(self.w_n2l),self.p_node_convT.assign(self.p_node_conv),self.h1_weightT.assign(self.h1_weight),self.h2_weightT.assign(self.h2_weight)]
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
        #self.session = tf.Session()

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.compat.v1.global_variables_initializer())

#################################################New code for graphDQN#####################################
    def BuildNet(self):
        # [2, embed_dim]
        w_n2l = tf.Variable(tf.compat.v1.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)
        # [embed_dim, embed_dim]
        p_node_conv = tf.Variable(tf.compat.v1.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.compat.v1.truncated_normal([2 * self.embedding_size, REG_HIDDEN], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim, 1]
            h2_weight = tf.Variable(tf.compat.v1.truncated_normal([REG_HIDDEN + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            #[reg_hidden + aux_dim, 1]
            last_w = h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.compat.v1.truncated_normal([2 * self.embedding_size, REG_HIDDEN], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        #[node_cnt, 2]
        nodes_size = tf.shape(self.n2nsum_param)[0]
        node_input = tf.ones((nodes_size,2))

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # no sparse
        input_message = tf.matmul(tf.cast(node_input,tf.float32), w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        input_potential_layer = tf.nn.relu(input_message)
        #input_potential_layer = input_message
        cdef int lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer

        while lv < max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            n2npool = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)
            #[node_cnt, embed_dim] * [embedding, embedding] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv)
            #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
            merged_linear = tf.add(node_linear,input_message)
            #[node_cnt, embed_dim]
            cur_message_layer = tf.nn.relu(merged_linear)

        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        y_potential = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
        #[[batch_size, embed_dim], [batch_size, embed_dim]] = [batch_size, 2*embed_dim], dense
        embed_s_a = tf.concat([action_embed,y_potential],1)
        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2 * embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1)
        #if reg_hidden == 0: ,[batch_size, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        loss = tf.losses.mean_squared_error(self.label, q_pred)
        trainStep = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # trainStep = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)
        #[[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
        embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden]
            last_output = tf.nn.relu(hidden)

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, w_n2l, p_node_conv, h1_weight, h2_weight

    def gen_graph(self, num_min, num_max):
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
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


    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        cdef double result_corehd = 0.0
        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            g_corehd = g.copy()
            result_corehd += self.CoreHD(g_corehd)
            self.InsertGraph(g, is_test=True)
        print ('Validation of CoreHD: %.16f'%(result_corehd / n_valid))


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
                    self.env_list[i].s0(TrainSet.Sample())
                    self.g_list[i] = self.env_list[i].graph

                    self.g_nx_list[i] = self.DeGenNetwork(self.g_list[i])
                    g_nx1 = self.g_nx_list[i]
                    Nodes_original1 = g_nx1.copy().nodes()
                    g_nx1_core = nx.k_core(g_nx1.copy(), 2)
                    Nodes_core1 = g_nx1_core.copy().nodes()
                    Nodes_removed1 = list(set(Nodes_original1) - set(Nodes_core1))
                    Nodes_removed1 = [int(node) for node in Nodes_removed1]
                    self.env_list[i].KcoreModify(Nodes_removed1, len(Nodes_core1))
                    self.g_nx_list[i] = g_nx1_core

            if n >= num_seq:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.Predict(self.g_list, [env.action_list for env in self.env_list], False)
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    ####Q1:argmax
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)

                g_nx2 = self.g_nx_list[i]
                Nodes_original2 = g_nx2.copy().nodes()
                g_nx2.remove_node(a_t)
                g_nx2_core = nx.k_core(g_nx2.copy(), 2)
                Nodes_core2 = g_nx2_core.copy().nodes()
                Nodes_removed2 = list(set(Nodes_original2) - set(Nodes_core2))
                Nodes_removed2 = [int(node) for node in Nodes_removed2]
                self.env_list[i].KcoreModify(Nodes_removed2, len(Nodes_core2))
                self.g_nx_list[i] = g_nx2_core

    #pass
    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['label'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph()
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat

    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph()
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
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
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })

            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                # cur_pred = []
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                        # cur_pred.append(-inf)
                    else:
                        cur_pred[k] = raw_output[pos]
                        # cur_pred.append(raw_output[pos])
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred


    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result


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
                double_list_pred = self.Predict(sample.g_list, sample.list_s_primes, False)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
                # print (double_list_predT)
                # print (list_pred)
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
        # list_target = list(np.zeros(BATCH_SIZE))
        # list_target = []
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

        return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

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
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.aux_input: np.array(self.inputs['aux_input']),
                                        self.label : self.inputs['label']})

            loss += result[0]*bsize
        return loss / len(g_list)
    #pass
    def Train(self):
        # VC = [] #record the vc of each 300 iterations
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

        save_dir = './models/Model_%d_%d'%(NUM_MIN, NUM_MAX)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')

        for iter in range(MAX_ITERATION):
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
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

    def Test(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_nx = self.DeGenNetwork(self.test_env.graph)
        nodes_num = nx.number_of_nodes(g_nx)

        g_nx1 = g_nx
        Nodes_original1 = g_nx1.copy().nodes()
        g_nx1_core = nx.k_core(g_nx1.copy(), 2)
        Nodes_core1 = g_nx1_core.copy().nodes()
        Nodes_removed1 = list(set(Nodes_original1) - set(Nodes_core1))
        Nodes_removed1 = [int(node) for node in Nodes_removed1]
        self.test_env.KcoreModify(Nodes_removed1, len(Nodes_core1))
        g_nx = g_nx1_core

        g_list.append(self.test_env.graph)
        cdef int cost = 0
        while (not self.test_env.isTerminal()):
            cost = cost + 1
            list_pred = self.Predict(g_list, [self.test_env.action_list], False)
            new_action = self.argMax(list_pred[0])
            self.test_env.step(new_action)

            g_nx2 = g_nx
            Nodes_original2 = g_nx2.copy().nodes()
            g_nx2.remove_node(new_action)
            g_nx2_core = nx.k_core(g_nx2.copy(), 2)
            Nodes_core2 = g_nx2_core.nodes()
            Nodes_removed2 = list(set(Nodes_original2) - set(Nodes_core2))
            Nodes_removed2 =[int(node) for node in Nodes_removed2]
            self.test_env.KcoreModify(Nodes_removed2, len(Nodes_core2))
            g_nx = g_nx2_core

        return cost / nodes_num

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

    def Evaluate(self, data_test,save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        cdef int i
        test_name = data_test.split('/')[-1]
        f = open(data_test, 'rb')
        result_file = '%s/test-%s-gnn-%s-%s.csv' % (save_dir, test_name, NUM_MIN, NUM_MAX)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            for i in tqdm(range(n_test)):
                g = cp.load(f)
                self.InsertGraph(g, is_test=True)
                t1 = time.time()
                val, sol = self.GetSol(i)
                t2 = time.time()
                #print('\n第%d个test图，vc:%.4f, time:%.4fs' % (i, val, t2 - t1))
                f_out.write('%.8f,' % val)
                #f_out.write('%d' % sol[0])
                for i in range(len(sol)):
                    f_out.write(' %d' % sol[i])
                f_out.write(',%.6f\n' % (t2 - t1))
                frac += val
                frac_time += (t2 - t1)
        print ('average size of vc: ', frac / n_test)
        print('average time: ', frac_time / n_test)
        return  frac / n_test,frac_time / n_test

    def GenNetworkWithIDMapping(self,g, node2id):  # networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            a2id = []
            b2id = []
            for i in range(len(edges)):
                a2id.append(node2id[a[i]])
                b2id.append(node2id[b[i]])
            A = np.array(a2id)
            B = np.array(b2id)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)

    def GetCoreDQNSolution(self, model_file, data_test_name, data_save_path, STEPRATIO):
        inner_env = mvc_env.py_MvcEnv(NUM_MAX)
        data_test_path = '../data/real/'
        ## begin computing...
        self.LoadModel(model_file)

        for dataName in data_test_name:
            print ('Calculating data: %s'%dataName)
            sol_decycle = []
            data_test = data_test_path + dataName + '.txt'
            data_save = data_save_path + dataName + '.txt'

            f_out = open(data_save,'w')
            ##read graph and do idMapping
            g_original = nx.read_edgelist(data_test)
            g_kcore, dmax, H, max_score_dict, score_dict, N0, N = self.preprocess(2, g_original)
            ## get decycle set
            done = False
            iteration = 0
            while N > 0 and done == False:
                ######################## GraphDQN dismantle ##############
                print ('Iteration: %d'%iteration)
                iteration += 1
                #do node id mapping
                Node2ID = {}
                ID2Node = {}
                numrealnodes = 0
                for node in g_kcore.nodes():
                    ID2Node[numrealnodes] = node
                    Node2ID[node] = numrealnodes
                    numrealnodes += 1
                #get kcore of graph
                #get inner graph with new idMapping
                g_kcore_inner = self.GenNetworkWithIDMapping(g_kcore, Node2ID)
                step = np.max([int(STEPRATIO*nx.number_of_nodes(g_kcore)),1]) #step size
                inner_env.s0(g_kcore_inner)
                ##do dqn attack and idMapping
                list_pred = self.Predict([g_kcore_inner], [inner_env.action_list], False)
                batchSol = np.argsort(-list_pred[0])[:step]
                for node in batchSol:
                    OriginNodeID = ID2Node[node]
                    sol_decycle.append(int(OriginNodeID))

                #################### update graph status ######################
                sol_len = 0
                while sol_len < len(batchSol) and len(g_kcore):
                    if max_score_dict[1] != None:   # 如果g_kcore中某些节点的度小于2，则先移除这些节点
                        d = 1
                        mx_scr_d = max_score_dict[d]
                        v = random.choice(list(H[d][mx_scr_d].keys()))
                    else:
                        v = ID2Node[batchSol[sol_len]]
                        sol_len += 1
                        if v in g_kcore.nodes():
                            d = max(g_kcore.degree(v), 1)
                            mx_scr_d = max_score_dict[d]
                        else:
                            continue

                    if v in g_kcore.nodes():
                        # remove element from H and set its score to None
                        del H[d][mx_scr_d][v]
                        score_dict[v] = None

                        # check for newest largest score
                        if H[d][mx_scr_d] == {}:	#判断度数为d的节点集合是否为空
                            del H[d][mx_scr_d]
                        # update max_score_dict
                        # suboptimal
                        try:
                            max_score_dict[d] = max(H[d].keys())
                        except ValueError:
                            max_score_dict[d] = None
                        ## update the neighbors
                        dv = g_kcore[v]
                        # remove neighbors from dict (for now)
                        for nb in dv:
                            self.remove_node_by_score(nb,g_kcore,H,max_score_dict,score_dict,2)
                        ## remove v from G
                        g_kcore.remove_node(v)
                        # update the degrees of the neighbors and their scores
                        for nb in dv:
                            self.add_node_by_score(nb,g_kcore,H,max_score_dict,score_dict,2)
                        ## check if dmax needs updating and do so if necessary. not necessary
                        if max_score_dict[d] == None:
                            # attempt to update
                            try:
                                while max_score_dict[dmax] == None:
                                    dmax -= 1
                            # unless there are no nodes left
                            except KeyError:
                                done = True
                                #sys.exit('Error!')
                ##################################################################
                N = len(g_kcore)

            sol_treebreak = TreeBreaker.TreeBreak(g_original, sol_decycle)
            sol = sol_decycle + sol_treebreak
            solution = sol
            for i in range(len(solution)):
                f_out.write('%d\n' % int(solution[i]))
                f_out.flush()
            f_out.close()


    def GetSol(self, int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef int cost = 0
        sol = []
        while (not self.test_env.isTerminal()):
            cost = cost + 1
            list_pred = self.Predict(g_list, [self.test_env.action_list],False)
            new_action = self.argMax(list_pred[0])
            self.test_env.step(new_action)
            sol.append(new_action)
        return cost / g_list[0].num_nodes, sol

    def SaveModel(self,model_path):
        self.saver.save(self.session, model_path)
        # print('model has been saved success!')

    def LoadModel(self,model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def loadGraphList(self, graphFileName):
        with open(graphFileName, 'rb') as f:
            graphList = cp.load(f)
        print("load graph file success!")
        return graphList

    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)

    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos

    def Max(self, scores):
        # print (scores)
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best


    def DeGenNetwork(self,pyGraph):
        G = nx.Graph()
        for edge in pyGraph.edge_list:
            G.add_edge(int(edge[0]),int(edge[1]))
        return G

    def CoreHD(self, g):
        nodeNum = nx.number_of_nodes(g)
        g_copy = g.copy()
        g_kcore = nx.k_core(g_copy, 2)
        sol = []
        while (nx.number_of_nodes(g_kcore)>0):
            dc = nx.degree_centrality(g_kcore)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(node)
            g_kcore.remove_node(node)
            g_kcore = nx.k_core(g_kcore, 2)
        Robustness = len(sol) / nodeNum
        return Robustness

    ##### copy from the code of CoreHD, some of them maybe useless here
    def score(self, v,G):
        return 0

    def category(self, v, k, G):
        return max(k-1,G.degree(v))

    def max_cat(self, k, G):
        # computes the max category in dict
        return max(dict(G.degree()).values())

    def preprocess(self, k,G):
        # suboptimal, but O(|G|), computation of some initial properties
        # size of initial graph
        N0 = len(G)
        # compute the k-core in O(|G|) steps (not really necessary)
        G = nx.k_core(G,k)
        # size of the initial k-core
        N = len(G)

        # if the core is empty we are done
        if N == 0:
            return G, 0, dict(), dict(), dict(), N0, N

        dmax = self.max_cat(k,G)

        # Initialize the dictionary, H, with
        # H[degree][score] = {i_1: 1, ..., i_r:1} and i_j indicating a node
        # degree = k-1,...,d because it is not necessary to distinguish nodes of degree smaller k.
        # the k-1 nodes need not be organized by score, but it makes the code less prune to errors and doesn't cost us much
        #H[k-1...dmax][]
        H = { d: dict() for d in range(k-1, dmax+1) }

        # collect the score for every node
        score_dict = {}
        # collect the max score for each 'degree' = k-1
        max_score_dict = {}

        for v in G.nodes():
            dgr = self.category(v,k,G)	#将k-core中度数小于k的节点度数标记为k-1,大于k的节点度数标记为度数本身
            scr = self.score(v,G)	#都为0，这里，HD
            score_dict[v] = scr
            # sort them in dictionaries by scores.
            try:
                H[dgr][scr][v] = 1
            except KeyError:
                # create if it does not already exist
                H[dgr][scr] = dict()
                H[dgr][scr][v] = 1

        # track currently largest score for each H[.]
        # H.keys() {1,2,3,4}
        for sub_dict_name in H.keys():
            try:
                mx_scr = max(H[sub_dict_name].keys())
            except ValueError:
                mx_scr = None
            max_score_dict[sub_dict_name] = mx_scr #字典，键值key范围：k-1,...,dmax，value都为0；如果若当前kcore图中没有节点的度数等于key，则max_score_dict[key]=none
        return G, dmax, H, max_score_dict, score_dict, N0, N


    def add_node_by_score(self, v,G,H,max_score_dict,score_dict,k):
        # get new score
        score_dict[v] = self.score(v,G)
        # get new category (degree) for dict reordering
        dgr = self.category(v,k,G)
        # insert into proper spot
        try:
            H[dgr][score_dict[v]][v] = 1
        except KeyError:
            # create if it does not already exist
            H[dgr][score_dict[v]] = dict()
            H[dgr][score_dict[v]][v] = 1
        # update max_score
        if max_score_dict[dgr] == None or max_score_dict[dgr] < score_dict[v]:
            max_score_dict[dgr] = score_dict[v]


    def remove_node_by_score(self, v,G,H,max_score_dict,score_dict,k):
        # find the right dict
        dgr = self.category(v,k,G)
        # remove the current nb
        del H[dgr][score_dict[v]][v]
        # if there are no more nodes of this particular score, remove the dict
        if H[dgr][score_dict[v]] == {}:
            del H[dgr][score_dict[v]]
        # if the score was equal to the max score, update max_score_dict
        if score_dict[v] == max_score_dict[dgr]:
            try:
                max_score_dict[dgr] = max(H[dgr].keys())
            # no more nodes left of degree d
            except ValueError:
                max_score_dict[dgr] = None
