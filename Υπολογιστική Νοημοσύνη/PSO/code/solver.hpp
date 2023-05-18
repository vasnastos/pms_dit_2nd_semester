#pragma once
#include "mlp_problem.hpp"
#include "pso.hpp"

#ifdef _WIN32
    const char sep='\\';
#else
    const char sep='/';
#endif


class Solver
{
    private:
        Problem *problem;
        PSO *pso;
        int units;
        Data best_weights;
        fs::path datasets_path;
        vector <string> dataset_names;

    public: 
        Solver(int number_of_units);
        ~Solver();

        void load(string filename);
        void flush();
        void solve();

        Data get_best_weights();
};