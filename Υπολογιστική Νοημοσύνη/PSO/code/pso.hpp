#pragma once
#include "problem.hpp"
#include "collection.hpp"
#include <fstream>

class PSO
{
    private:
        Problem *problem;
        Collection particle;
        Collection best_particle;
        Collection velocity;

        int particle_count;
        double inertia;
        double inertia_max,inertia_min;
        size_t iter;
        size_t max_iters;
        Data best_x;
        double best_y;
        Data y_distribution;
        
        int c1,c2,c3;
        mt19937 eng;

        bool terminated();
        void step();

    public:
        PSO(Problem *p,int number_of_iterations=1000,int particle_count=500);
        ~PSO();

        void set_max_iters(int number_of_iterations);
        int get_max_iters()const;

        void set_particle_count(int cnt);
        int get_particle_count()const;

        Data get_best_x();
        double get_best_y();
        Data geometric_center();
        
        void solve();
        void save_y();
};