#include <iostream>
#include "XGBoost.h"

LightGBM::CovTypeClassifier clf;
int main() {
    std::cout << "Hello, World!" << std::endl;
    float values[4] = {0.644, 0.247, -0.447, 0.862};
    float result = clf.predict(values);
    printf("Result: %f\n", result);
    return 0;
}
