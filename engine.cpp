#include "engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

Value::Value(float data, unordered_set<Value*> prev, std::string op)
    : data(data), grad(0.0), prev(move(prev)), op(move(op)) {
    
    _backward = []() {}; 
}

float Value::get_data() const { return data; }
void Value::set_data(float data) { this->data = data; }
float Value::get_grad() const { return grad; }
void Value::set_grad(float grad_value) { this->grad = grad_value; }


string Value::get_op() const { return op; } 

unordered_set<Value*> Value::get_prev() const { return prev; }

Value* Value::operator+(Value* other) {
    auto out_prev = unordered_set<Value*>{this, other};
    Value* out = new Value(this->data + other->data, out_prev, "+");

    out->_backward = [this, other, out]() {
        this->grad += out->grad;
        other->grad += out->grad;
    };
    return out;
}

Value* Value::operator-() {
    
    float out_data = -this->data;
    auto out_prev = unordered_set<Value*>{this};
    Value* out = new Value(out_data, out_prev, "neg");
    
    out->_backward = [this, out]() {
        this->grad += -1.0f * out->grad;
    };
    return out;
}


Value* Value::operator-(Value* other) {
    
    Value* minus_one = new Value(-1.0f); 
    
    
    Value* neg_other = other->operator*(minus_one); 
    
    
    Value* result = this->operator+(neg_other); 
    
    result->op = "-";
    
    
    return result; 
}

Value* Value::operator*(Value* other) {
    auto out_prev = unordered_set<Value*>{this, other};
    Value* out = new Value(this->data * other->data, out_prev, "*");

    out->_backward = [this, other, out]() {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}


Value* Value::operator/(Value* other) {
    
    Value* minus_one = new Value(-1.0f); 
    Value* inverse = other->pow(minus_one); 
    Value* result = this->operator*(inverse);
    result->op = "/";
    
    return result;
}

Value* Value::pow(Value* other) {
    float out_data = std::pow(this->data, other->data);
    auto out_prev = unordered_set<Value*>{this, other};
    Value* out = new Value(out_data, out_prev, "^");

    out->_backward = [this, other, out]() {
        
        this->grad += other->data * std::pow(this->data, other->data - 1.0f) * out->grad;
        
    };
    return out;
}

Value* Value::relu() {
    float out_data = this->data < 0 ? 0.0f : this->data;
    auto out_prev = unordered_set<Value*>{this};
    Value* out = new Value(out_data, out_prev, "ReLU");
    
    out->_backward = [this, out]() {
        this->grad += (out->data > 0.0f ? 1.0f : 0.0f) * out->grad;
    };
    return out;
}

Value* Value::tanh() {
    float out_data = std::tanh(this->data);
    auto out_prev = unordered_set<Value*>{this};
    Value* out = new Value(out_data, out_prev, "tanh");
    
    out->_backward = [this, out_data, out]() {
        
        float tanh_sq = out_data * out_data;
        this->grad += (1.0f - tanh_sq) * out->grad;
    };
    return out;
}

void Value::backward() {
    vector<Value*> topo;
    unordered_set<Value*> visited;

    function<void(Value*)> build_topo = [&](Value* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto child : v->get_prev()) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(this);
    grad = 1.0;

    
    reverse(topo.begin(), topo.end());

    
    for (auto v : topo) {
        v->_backward();
    }
}



Value* operator+(Value& lhs, Value& rhs) { return lhs.operator+(&rhs); }
Value* operator-(Value& lhs, Value& rhs) { return lhs.operator-(&rhs); }
Value* operator*(Value& lhs, Value& rhs) { return lhs.operator*(&rhs); }
Value* operator/(Value& lhs, Value& rhs) { return lhs.operator/(&rhs); }
Value* pow(Value& lhs, Value& rhs) { return lhs.pow(&rhs); }
