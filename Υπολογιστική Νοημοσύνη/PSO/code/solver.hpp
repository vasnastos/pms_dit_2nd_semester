#pragma once
#include "mlp_problem.hpp"
#include "pso.hpp"


class Solver
{
    private:
        Problem *problem;
        PSO *pso;
        Data best_weights;
    public: 
        Solver();
        ~Solver();

        void load(string filename);
        void flush();
        void solve();

        Data get_best_weights();
};