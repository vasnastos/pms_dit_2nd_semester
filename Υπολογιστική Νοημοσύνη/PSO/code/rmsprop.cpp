#include "rmsprop.hpp"

RMSPROP::RMSPROP(Problem *instance):problem(instance) {
    this->squared_gradients.resize(instance->get_dimension());
}

RMSPROP::~RMSPROP() {}

void RMSPROP::solve() {

}

void RMSPROP::save(string filename)
{

}   