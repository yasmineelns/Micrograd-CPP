#include <iostream>
#include <fstream>
#include <vector>
#include "engine.h"
#include "nn.h"
#include "graph.h"

using namespace std;

void run_mlp_test() {
    cout << "--- C++ Micrograd MLP Test ---" << endl;

    // 1️⃣ Création d'un réseau : 3 entrées → 2 couches cachées de 4 neurones → 1 sortie
    MLP model(3, {4, 4, 1});

    // 2️⃣ Création des entrées
    vector<Value*> x = {
        new Value(1.0f, {}, "x1"),
        new Value(-2.0f, {}, "x2"),
        new Value(0.5f, {}, "x3")
    };

    // 3️⃣ Forward pass
    vector<Value*> y_pred = model(x);
    cout << "\nOutput y: " << y_pred[0]->get_data() << endl;

    // 4️⃣ Backward pass
    y_pred[0]->backward();
    cout << "\n--- Gradients ---" << endl;

    // Gradients des entrées
    for (Value* xi : x)
        cout << xi->get_op() << " grad: " << xi->get_grad() << endl;

    // Gradients des paramètres du modèle
    int param_idx = 0;
    for (Value* p : model.parameters()) {
        cout << "Param[" << param_idx++ << "] data: " << p->get_data() << " | grad: " << p->get_grad() << endl;
    }

    // 5️⃣ Génération du graphe computationnel
    string dot_code = generate_dot(y_pred[0]);
    const string filename = "mlp_graph.dot";
    ofstream file(filename);
    if (file.is_open()) {
        file << dot_code;
        file.close();
        cout << "\nComputation graph generated and saved to: " << filename << endl;
        cout << "To visualize, use Graphviz (e.g., 'dot -Tpng " << filename << " -o mlp_graph.png')" << endl;
    } else {
        cerr << "\nError: Could not open file " << filename << " for writing the DOT graph." << endl;
    }

    // Libération de la mémoire
    for (Value* xi : x) delete xi;
}

int main() {
    run_mlp_test();
    return 0;
}
