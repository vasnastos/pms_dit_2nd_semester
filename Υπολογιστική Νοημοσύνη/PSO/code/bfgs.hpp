#pragma once
#include "problem.hpp"


class BFGS
{
    private:
        Problem *problem;
        Data xpoint;
        Data gradients;
        double ypoint;
        int max_iters;
        int iter_id;

        vector <double> Hessian;

        double alpha;
        double beta;
        double tolerance;

        void step();
        bool termination();

        double norm_grad();
    public:
        BFGS(Problem *in_problem,Data &initial_guess,int max_iters);
        void solve();

        Data get_best_x();
};