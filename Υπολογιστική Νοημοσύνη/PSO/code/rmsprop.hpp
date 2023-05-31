#pragma once
#include "problem.hpp"

class RMSPROP
{
    private:
        Problem *problem;
        Data xpoint;
        double objective_value;
        double decay_rate;//rho
        double learning_rate;
        Data squared_gradients;

        Data y_distribution;
    public:
        RMSPROP(Problem *instance);
        ~RMSPROP();

        void solve();
        void save(string filename);
        Data get_best_x()const;
};