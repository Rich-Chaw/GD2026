#ifndef PREPAREBATCHGRAPH_H_
#define PREPAREBATCHGRAPH_H_

#include "graph.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>

class sparseMatrix
{
 public:
    sparseMatrix();
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class PrepareBatchGraph{
public:
    PrepareBatchGraph();
    void SetupGraphInput(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions);
    void SetupTrain(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions);
    void SetupPredAll(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered);
    int GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, int& counter, std::vector<int>& idx_map);
    std::shared_ptr<sparseMatrix> act_select;
    std::shared_ptr<sparseMatrix> rep_global;
    std::shared_ptr<sparseMatrix> n2nsum_param;
    std::shared_ptr<sparseMatrix> subgsum_param;
    std::vector< std::vector<int> > idx_map_list;
    std::vector< std::vector<double> > aux_feat;
    GraphStruct graph;
    std::vector<int> avail_act_cnt;
};

std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph);
#endif