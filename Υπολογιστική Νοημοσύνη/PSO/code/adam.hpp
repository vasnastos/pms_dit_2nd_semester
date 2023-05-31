#pragma once
#include "problem.hpp"


class Adam
{
    // m=>first momentum
    // u=>second momentum
    private:
        Problem *problem;
        double alpha,beta1,beta2;
        Data m,u,mhat,uhat;
        double learning_rate;
        Data xpoint;
        double objective_value;

        Data y_distribution;

    public:
        Adam(Problem *instance);
        ~Adam();

        void solve();

        void set_alpha(double new_alpha_value);
        double get_alpha()const;
        void set_beta1(double new_beta1_value);
        double get_beta1()const;
        void set_beta2(double new_beta2_value);
        double get_beta2()const;
        void set_learning_rate(double new_epsilon);
        double get_learning_rate()const;
        Data get_best_x()const;

        void save(string filename);

};
