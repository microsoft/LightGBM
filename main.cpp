#include <iostream>
#include "Predict.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;
    float values[4] = {0.1, 0.2, 0.3, 0.4};
    float result = predict(values);
    printf("Result: %f\n", result);
    return 0;
}
