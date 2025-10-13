#include "engine.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;


void test_sanity_check() {
    cout << "\n=== TEST SANITY CHECK ===\n";
    
    Value* x = new Value(-4.0);
    
    // z = 2 * x + 2 + x
    Value* two = new Value(2.0);
    Value* temp1 = (*two) * (*x);      // 2 * x = -8
    Value* temp2 = (*temp1) + (*two);  // -8 + 2 = -6
    Value* z = (*temp2) + (*x);        // -6 + (-4) = -10
    
    // q = z.relu() + z * x
    // Note: ReLU(-10) = 0
    Value* z_relu = z->relu();         // 0
    Value* z_times_x = (*z) * (*x);    // -10 * -4 = 40
    Value* q = (*z_relu) + (*z_times_x); // 0 + 40 = 40
    
    // h = (z * z).relu()
    Value* z_squared = (*z) * (*z);    // -10 * -10 = 100
    Value* h = z_squared->relu();      // 100
    
    // y = h + q + q * x
    Value* temp3 = (*h) + (*q);        // 100 + 40 = 140
    Value* q_times_x = (*q) * (*x);    // 40 * -4 = -160
    Value* y = (*temp3) + (*q_times_x); // 140 + (-160) = -20
    
    y->backward();
    
    cout << "y.data = " << y->get_data() << " (attendu: -20.0)\n";
    cout << "x.grad = " << x->get_grad() << " (attendu: 46.0)\n";
    

    assert(abs(y->get_data() - (-20.0)) < 1e-6);
    assert(abs(x->get_grad() - 46.0) < 1e-6);
    
    cout << "✓ Test sanity check PASSE!\n";
}

// Test avec plus d'opérations 
void test_more_ops() {
    cout << "\n=== TEST MORE OPS ===\n";
    
    Value* a = new Value(-4.0);
    Value* b = new Value(2.0);
    
    // c = a + b
    Value* c = (*a) + (*b);  // -2
    
    // d = a * b + b**3
    Value* three = new Value(3.0);
    Value* a_times_b = (*a) * (*b);     // -8
    Value* b_cubed = b->pow(three);     // 8
    Value* d = (*a_times_b) + (*b_cubed); // 0
    
    // c += c + 1
    Value* one = new Value(1.0);
    Value* c_plus_c = (*c) + (*c);      // -4
    c = (*c_plus_c) + (*one);           // -3
    
    // c += 1 + c + (-a)
    Value* one_plus_c = (*one) + (*c);  // -2
    Value* neg_a = -(*a);               // 4
    Value* temp = (*one_plus_c) + (*neg_a); // 2
    c = (*c) + (*temp);                 // -1
    
    // d += d * 2 + (b + a).relu()
    Value* two = new Value(2.0);
    Value* d_times_2 = (*d) * (*two);   // 0
    Value* b_plus_a = (*b) + (*a);      // -2
    Value* relu1 = b_plus_a->relu();    // 0
    Value* temp2 = (*d_times_2) + (*relu1); // 0
    d = (*d) + (*temp2);                // 0
    
    // d += 3 * d + (b - a).relu()
    Value* three_times_d = (*three) * (*d); // 0
    Value* b_minus_a = (*b) - (*a);     // 6
    Value* relu2 = b_minus_a->relu();   // 6
    Value* temp3 = (*three_times_d) + (*relu2); // 6
    d = (*d) + (*temp3);                // 6
    
    // e = c - d
    Value* e = (*c) - (*d);             // -7
    
    // f = e**2
    Value* f = e->pow(two);             // 49
    
    // g = f / 2.0
    Value* g = (*f) / (*two);           // 24.5
    
    // g += 10.0 / f
    Value* ten = new Value(10.0);
    Value* ten_div_f = (*ten) / (*f);   // 0.204...
    g = (*g) + (*ten_div_f);            // 24.7...
    
    g->backward();
    
    cout << "g.data = " << g->get_data() << " (attendu: 24.7041...)\n";
    cout << "a.grad = " << a->get_grad() << " (attendu: 138.8338...)\n";
    cout << "b.grad = " << b->get_grad() << " (attendu: 645.5772...)\n";
    
    
    double tol = 1e-4;
    assert(abs(g->get_data() - 24.70408163265306) < tol);
    assert(abs(a->get_grad() - 138.83381924198252) < tol);
    assert(abs(b->get_grad() - 645.5772594752186) < tol);
    
    cout << "✓ Test more ops PASSE!\n";
}

int main() {
    cout << "\n╔══════════════════════════════════════════╗\n";
    cout << "║  TESTS DE VERIFICATION AUTODIFF ENGINE  ║\n";
    cout << "╚══════════════════════════════════════════╝\n";
    
    try {
        test_sanity_check();
        test_more_ops();
        
        cout << "\n" << string(44, '=') << "\n";
        cout << "✓✓✓ TOUS LES TESTS PASSENT! ✓✓✓\n";
        cout << string(44, '=') << "\n\n";
    } catch (const exception& e) {
        cout << "\n✗ ERREUR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}