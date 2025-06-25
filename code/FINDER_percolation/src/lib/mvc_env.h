#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"

class MvcEnv
{
public:
    MvcEnv(double _norm);

    void s0(std::shared_ptr<Graph> _g);

    void KcoreModify(std::vector<int> nodeOutofKCore,int _KcoreNodesNum);

    int KcoreNodesNum;

    double step(int a);

    int randomAction();

    bool isTerminal();

    double getReward();

    double norm;

    std::shared_ptr<Graph> graph;

    std::vector< std::vector<int> > state_seq;

    std::vector<int> act_seq, action_list;

    std::vector<double> reward_seq, sum_rewards;

    int numCoveredEdges;

    std::set<int> covered_set;

    std::vector<int> avail_list;
};

#endif