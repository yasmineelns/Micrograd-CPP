#include "nn.h"
#include <cmath>
#include <iostream>
#include <cstdlib> 
#include <numeric>


static double rand_double(double low, double high) {

    return low + (high - low) * (double(rand()) / RAND_MAX);
}


Neuron::Neuron(int nin) :b(rand_double(-1.0, 1.0)){

    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }
    
    for (int i = 0; i < nin; i++) {

        w.emplace_back(rand_double(-1.0, 1.0));
    }
}

Value* Neuron::operator()(std::vector<Value*>& x) {
    if (x.size() != w.size()) {
        std::cerr << "Error: Input size mismatch for Neuron." << std::endl;
        return nullptr;
    }


    Value* act_ptr = nullptr;
    
    for (size_t i = 0; i < w.size(); i++) {

        Value* term = (*x[i]) * w[i]; 
        
        if (i == 0) {
            act_ptr = term;
        } else {

            Value* new_act = (*act_ptr) + (*term); 

            act_ptr = new_act;
        }
    }
    

    Value* new_act_ptr = (*act_ptr) + b; 
    

    Value* out = new_act_ptr->tanh();


    return out;
}

std::vector<Value*> Neuron::parameters() {
    std::vector<Value*> params;
    for (Value& val : w) {
        params.push_back(&val);
    }
    params.push_back(&b);
    return params;
}


Layer::Layer(int nin, int nout) {
    for (int i = 0; i < nout; i++) {
        neurons.emplace_back(nin);
    }
}

std::vector<Value*> Layer::operator()(std::vector<Value*>& x) {
    std::vector<Value*> outs;
    for (Neuron& n : neurons) {
        outs.push_back(n(x));
    }
    return outs;
}

std::vector<Value*> Layer::parameters() {
    std::vector<Value*> params;
    for (Neuron& n : neurons) {
        std::vector<Value*> n_params = n.parameters();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}


MLP::MLP(int nin, std::vector<int> nouts) {
    int size = nin;
    for (int nout : nouts) {
        layers.emplace_back(size, nout);
        size = nout;
    }
}

std::vector<Value*> MLP::operator()(std::vector<Value*>& x) {
    std::vector<Value*> inputs = x;
    for (Layer& l : layers) {
        inputs = l(inputs);
    }
    return inputs;
}

std::vector<Value*> MLP::parameters() {
    std::vector<Value*> params;
    for (Layer& l : layers) {
        std::vector<Value*> l_params = l.parameters();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}
