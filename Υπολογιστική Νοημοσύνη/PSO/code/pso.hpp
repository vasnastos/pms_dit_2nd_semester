#pragma once
#include "problem.hpp"
#include <fstream>
#ifdef __linux__
#include <climits>
#endif

struct Particle
{
    static Data left_bound;
    static Data right_bound;
    
    Data velocity;
    Data position;
    Data best_position;
    double best_objective;
    double objective_value;

    Particle();
    Particle(Data &x);
    Particle(Data &x,double &obj_value);

    void update_velocity(double &inertia,double &c1,double &c2,Particle &global_best);
    bool update_position();
    void update_objective_value(double &new_obj_value,Particle &global_best_particle);
};


class PSO
{
    private:
        Problem *problem;
        vector <Particle> particles;
        Particle best_particle;
        Particle old_best_particle;

        double inertia;
        double c1,c2;
        double inertia_max,inertia_min;
        int iter,max_iters,num_particles;

        Data objectives;
        int k;
        double midpoint;

        void update_inertia();
        void step();
        bool terminated();
        void get_best_worst_objective(double &max_obj,double &min_obj);

        // For stopping rule
        double v1,v2; // auxiliary variables
        double variance;

    public:
        PSO(Problem *data,int num_particles,int num_iterations);
        void solve();
        void save(string filename);

        Data get_best_x();
};