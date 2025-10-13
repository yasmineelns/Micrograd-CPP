#ifndef NN_H
#define NN_H

#include "engine.h"
#include <vector>
#include <random>
#include <ctime>


class Neuron {
public:
    
    std::vector<Value> w;   
    Value b;                

    
    Neuron(int nin);

    
    Value* operator()(std::vector<Value*>& x);
    
    
    std::vector<Value*> parameters();
};


class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout);

    
    std::vector<Value*> operator()(std::vector<Value*>& x);
    
    std::vector<Value*> parameters();
};


class MLP {
public:
    std::vector<Layer> layers;

    MLP(int nin, std::vector<int> nouts);

    
    std::vector<Value*> operator()(std::vector<Value*>& x);

    std::vector<Value*> parameters();
};

#endif
