#include "collection.hpp"
#include <algorithm>
#ifdef __linux__
    #include <climits>
#endif
#include "fuzzyset.hpp"

struct APSOhyperparameters
{
    double inertia_max,inertia_min;
    double c1,c2;
    double constrictor_factor,ff;
    double elitist_learning_rate_min,elitist_learning_rate_max;

    APSOhyperparameters();
    APSOhyperparameters(double in_max,double in_min,double c1_param,double c2_param,double cf,double ff_param,double elrmn,double elrmx);
};


class APSO
{
    private:
        Problem *problem;
        APSOhyperparameters params;
        
        Collection particles;
        Collection velocity;
        Collection best_particle;



        FuzzySet *adaptive_set;
        int previous_state;

        size_t iter_id;
        size_t global_best_particle_idx;
        size_t max_iters;
        size_t particle_count;
        double inertia;
        double elitist_learning_rate;
        mt19937 eng;
     
        Data best_x;
        double best_y;

        void step();
        bool terminated();
        void adjust_acceleration_coefficients(double &acceleration_rate,string c1_direction,string c2_direction);
        void print_global_best();

    public:
        APSO(Problem *p,int num_particles,int max_iters);
        ~APSO();

        void solve();

        // ESE procedure
        double eucleidian_distance(Data &particle_1,Data &particle_2);
        double particle_mean_distance(int particle_idx);
        double evaluationary_factor();
        void ESE();
        void ELS(); //elitist learning operation

        Data get_best_x()const;
        double get_best_y()const;
};