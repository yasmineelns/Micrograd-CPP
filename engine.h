#ifndef ENGINE_H
#define ENGINE_H

#include <functional>
#include <unordered_set>
#include <string>
#include <vector>
#include <ostream>

using namespace std;

class Value {
public:
    float data;
    float grad;
    function<void()> _backward;
    unordered_set<Value*> prev;
    string op;

public:
    Value(float data, unordered_set<Value*> prev = {}, string op = "");
    ~Value() = default;
    void set_grad(float grad_value);
    float get_data() const;
    void set_data(float data);
    float get_grad() const;
    string get_op() const;
    unordered_set<Value*> get_prev() const;

    Value* operator+(Value* other);
    Value* operator-(Value* other);
    Value* operator/(Value* other);
    Value* operator*(Value* other);
    Value* pow(Value* other);
    Value* operator-();

    void backward();
    Value* relu();
    Value* tanh();
    friend Value* operator+(Value& lhs, Value& rhs);
    friend Value* operator-(Value& lhs, Value& rhs);
    friend Value* operator*(Value& lhs, Value& rhs);
    friend Value* operator/(Value& lhs, Value& rhs);
    friend Value* pow(Value& lhs, Value& rhs);
};

// Non-member operators
Value* operator+(Value& lhs, Value& rhs);
Value* operator-(Value& lhs, Value& rhs);
Value* operator*(Value& lhs, Value& rhs);
Value* operator/(Value& lhs, Value& rhs);
Value* pow(Value& lhs, Value& rhs);

#endif
