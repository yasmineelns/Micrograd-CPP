#include "graph.h"
#include <sstream>
#include <unordered_set>
#include <iomanip>
#include <vector>
#include <utility> 
#include <cstdint> 

using namespace std;

/**
 * @brief Walks the computation graph and collects all nodes and edges.
 */
static void build_graph(Value* v, unordered_set<Value*>& nodes, vector<pair<Value*, Value*>>& edges) {
    if (nodes.find(v) == nodes.end()) {
        nodes.insert(v);

        for (Value* child : v->get_prev()) {
            edges.emplace_back(child, v); 
            build_graph(child, nodes, edges);
        }
    }
}

std::string generate_dot(Value* root) {
    if (!root) return "digraph G { label=\"Empty Graph\" }";
    
    unordered_set<Value*> nodes;
    vector<pair<Value*, Value*>> edges;
    build_graph(root, nodes, edges);

    stringstream ss;
    ss << "digraph G {" << endl;
    ss << "  rankdir=\"LR\";" << endl;
    ss << "  node [shape=record, style=\"filled\", fillcolor=\"#f9f9f9\", fontname=\"Arial\"];" << endl;
    ss << "  edge [color=\"#999999\"];" << endl;

    
    for (Value* v : nodes) {
        
        string node_id = "v" + to_string((uintptr_t)v);
        
        stringstream label_ss;
        label_ss << fixed << setprecision(4);
        
        
        if (!v->get_op().empty()) {
            label_ss << "{" << v->get_op() << " | data " << v->get_data() << " | grad " << v->get_grad() << "}";
        } else {
            label_ss << "{" << v->get_data() << " | grad " << v->get_grad() << "}";
        }
        
        ss << "  " << node_id << " [label=\"" << label_ss.str() << "\"];" << endl;
    }

    ss << "  // Connections (edges)" << endl;

    
    for (const auto& edge : edges) {
        Value* n1 = edge.first; 
        Value* n2 = edge.second; 
        
        string n1_id = "v" + to_string((uintptr_t)n1);
        string n2_id = "v" + to_string((uintptr_t)n2);
        
        
        ss << "  " << n1_id << " -> " << n2_id << ";" << endl;
    }
    
    ss << "}" << endl;
    return ss.str();
}
