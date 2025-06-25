#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>

MvcEnv::MvcEnv(double _norm)
{
norm = _norm;
KcoreNodesNum = _norm;
graph = nullptr;
numCoveredEdges = 0;
state_seq.clear();
act_seq.clear();
action_list.clear();
reward_seq.clear();
sum_rewards.clear();
covered_set.clear();
avail_list.clear();
}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
//    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

void MvcEnv::KcoreModify(std::vector<int> nodeOutofKCore,int _KcoreNodesNum){
    for (auto a : nodeOutofKCore)
    {
        covered_set.insert(a);
        action_list.push_back(a);
    }

    KcoreNodesNum = _KcoreNodesNum;
}

double MvcEnv::step(int a)
{
    assert(graph);
    assert(covered_set.count(a) == 0);
//    action_list.push_back(a);
    state_seq.push_back(action_list);
    act_seq.push_back(a);

//    covered_set.insert(a);
//    action_list.push_back(a);

    double r_t = getReward();
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);
    return r_t;
}



int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
        {
            bool useful = false;
            for (auto neigh : graph->adj_list[i])
                if (covered_set.count(neigh) == 0)
                {
                    useful = true;
                    break;
                }
            if (useful)
                avail_list.push_back(i);
        }
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}


bool MvcEnv::isTerminal()
{
//    assert(graph);
//    return graph->num_edges == numCoveredEdges;
    return KcoreNodesNum == 0;
}

double MvcEnv::getReward()
{
    return -1.0 / norm;
}