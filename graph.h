#ifndef GRAPH_VIZ_H
#define GRAPH_VIZ_H

#include "engine.h"
#include <string>

/**
 * @brief Generates a DOT graph representation of the computation graph 
 * starting from the given Value pointer.
 * * The output string can be saved to a file (e.g., 'graph.dot') and 
 * visualized using Graphviz (e.g., 'dot -Tpng graph.dot -o graph.png').
 * * @param root The output Value* to start the traversal from.
 * @return std::string The DOT graph definition.
 */
std::string generate_dot(Value* root);

#endif 
