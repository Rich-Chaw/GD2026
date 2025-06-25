#include "PrepareBatchGraph.h"

sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
}

 PrepareBatchGraph::PrepareBatchGraph()
{

}

int PrepareBatchGraph::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered,int& counter, std::vector<int>& idx_map)
{
    std::set<int> c;

    idx_map.resize(g->num_nodes);

    for (int i = 0; i < g->num_nodes; ++i)
        idx_map[i] = -1;

    for (int i = 0; i < num; ++i)
        c.insert(covered[i]);  

    counter = 0;

    int n = 0;

    for (auto& p : g->edge_list)
    {

        if (c.count(p.first) || c.count(p.second))
        {
            counter++;         
        } else {

            if (idx_map[p.first] < 0)
                n++;

            if (idx_map[p.second] < 0)
                n++;
            idx_map[p.first] = 0;
            idx_map[p.second] = 0;
        }
    }    
    return n;
}

void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list, 
                           std::vector< std::vector<int> > covered, 
                           const int* actions)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());

    idx_map_list.resize(idxes.size());
    avail_act_cnt.resize(idxes.size());

    int node_cnt = 0;

    for (size_t i = 0; i < idxes.size(); ++i)
    {   
        std::vector<double> temp_feat;

        auto g = g_list[idxes[i]];

        int counter;

        if (g->num_nodes)
            temp_feat.push_back((double)covered[idxes[i]].size() / (double)g->num_nodes);

        avail_act_cnt[i] = GetStatusInfo(g, covered[idxes[i]].size(), covered[idxes[i]].data(), counter, idx_map_list[i]);

        if (g->edge_list.size())
            temp_feat.push_back((double)counter / (double)g->edge_list.size());

        temp_feat.push_back(1.0);

        node_cnt += avail_act_cnt[i];
        aux_feat.push_back(temp_feat);
    }

    graph.Resize(idxes.size(), node_cnt);

    if (actions)
    {
        act_select->rowNum=idxes.size();
        act_select->colNum=node_cnt;
    } else
    {
        rep_global->rowNum=node_cnt;
        rep_global->colNum=idxes.size();
    }


    node_cnt = 0;
    int edge_cnt = 0;

    for (size_t i = 0; i < idxes.size(); ++i)
    {             
        auto g = g_list[idxes[i]];
        auto idx_map = idx_map_list[i];

        int t = 0;
        for (int j = 0; j < g->num_nodes; ++j)
        {   
            if (idx_map[j] < 0)
                continue;
            idx_map[j] = t;
            graph.AddNode(i, node_cnt + t);
            if (!actions)
            {
                rep_global->rowIndex.push_back(node_cnt + t);
                rep_global->colIndex.push_back(i);
                rep_global->value.push_back(1.0);
            }
            t += 1;
        }
        assert(t == avail_act_cnt[i]);

        if (actions)
        {   
            auto act = actions[idxes[i]];
            assert(idx_map[act] >= 0 && act >= 0 && act < g->num_nodes);
            act_select->rowIndex.push_back(i);
            act_select->colIndex.push_back(node_cnt + idx_map[act]);
            act_select->value.push_back(1.0);
        }
        
        for (auto p : g->edge_list)
        {   
            if (idx_map[p.first] < 0 || idx_map[p.second] < 0)
                continue;
            auto x = idx_map[p.first] + node_cnt, y = idx_map[p.second] + node_cnt;
            graph.AddEdge(edge_cnt, x, y);
            edge_cnt += 1;
            graph.AddEdge(edge_cnt, y, x);
            edge_cnt += 1;
        }
        node_cnt += avail_act_cnt[i];
    }
    assert(node_cnt == (int)graph.num_nodes);

    n2nsum_param = n2n_construct(&graph);
    subgsum_param = subg_construct(&graph);

}


void PrepareBatchGraph::SetupTrain(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions)
{
    SetupGraphInput(idxes, g_list, covered, actions);
}


void PrepareBatchGraph::SetupPredAll(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered)
{
    SetupGraphInput(idxes, g_list, covered, nullptr);
}



std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;
	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];

		for (size_t j = 0; j < list.size(); ++j)
		{

            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].second);
		}
	}
    return result;
}


std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;
	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}



std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;

	for (unsigned int i = 0; i < graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (unsigned int i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from];
        for (size_t j = 0; j < list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph;
    result->colNum = graph->num_nodes;
	for (unsigned int i = 0; i < graph->num_subgraph; ++i)
	{
		auto& list = graph->subgraph->head[i];

		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
	}
    return result;
}
