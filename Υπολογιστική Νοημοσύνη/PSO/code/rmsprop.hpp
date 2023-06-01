#pragma once
#include "problem.hpp"

class RMSPROP
{
    private:
        Problem *problem;
        Data xpoint;
        double objective_value;
        double rho;//decay rate
        Data learning_rate;
        double epsilon;
        Data squared_gradients;

        Data y_distribution;
    public:
        RMSPROP(Problem *instance);
        ~RMSPROP();

        void solve();
        void save(string filename);
        Data get_best_x()const;
};