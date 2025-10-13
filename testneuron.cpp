#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "engine.h"
#include "nn.h"
#include "graph.h" 

using namespace std;

/**
 * @brief Runs the single Neuron test, computes gradients, and generates the graph file.
 */
void run_neuron_test() {
    cout << "--- C++ Micrograd Neuron Test ---" << endl;
    

    srand(static_cast<unsigned int>(time(nullptr)));
    

    Neuron n(2);
    

    Value* x1 = new Value(1.0f, {}, "x1");
    Value* x2 = new Value(-2.0f, {}, "x2");
    vector<Value*> x = {x1, x2};
    

    cout << "Neuron parameters (randomly initialized):" << endl;
    cout << "w[0]: " << n.w[0].get_data() << " | b: " << n.b.get_data() << endl;
    cout << "w[1]: " << n.w[1].get_data() << endl;
    cout << "\nInput x: [data=1.0, grad=" << x1->get_grad() << "] and [data=-2.0, grad=" << x2->get_grad() << "]" << endl;


    Value* y = n(x);

    if (y == nullptr) {
        cerr << "Forward pass failed. Exiting." << endl;
        delete x1;
        delete x2;
        return;
    }
    
    cout << "Output y (before backward): " << y->get_data() << endl;

    y->backward();
    cout << "\nBackward pass completed." << endl;
    cout << "--- Final Gradients ---" << endl;
    cout << "Output y: " << y->get_data() << " (grad: " << y->get_grad() << ")" << endl;
    cout << "Input x1: " << x1->get_data() << " (grad: " << x1->get_grad() << ")" << endl;
    cout << "Input x2: " << x2->get_data() << " (grad: " << x2->get_grad() << ")" << endl;
    cout << "w[0]: " << n.w[0].get_data() << " (grad: " << n.w[0].get_grad() << ")" << endl;
    cout << "w[1]: " << n.w[1].get_data() << " (grad: " << n.w[1].get_grad() << ")" << endl;
    cout << "b:    " << n.b.get_data() << " (grad: " << n.b.get_grad() << ")" << endl;

    string dot_code = generate_dot(y);
    const string filename = "neuron_graph.dot";
    
    ofstream file(filename);
    if (file.is_open()) {
        file << dot_code;
        file.close();
        cout << "\nComputation graph generated and saved to: " << filename << endl;
        cout << "To visualize, use Graphviz (e.g., 'dot -Tpng " << filename << " -o graph.png')" << endl;
    } else {
        cerr << "\nError: Could not open file " << filename << " for writing the DOT graph." << endl;
    }

    delete x1;
    delete x2;
}

int main() {
    run_neuron_test();
    return 0;
}
