#include "collection.hpp"
#ifdef __linux__
    #include <climits>
#endif


class APSO
{
    private:
        Problem *problem;

        Collection particles;
        Collection velocity;
        Collection best_particle;

        int iter_id;
        int max_iters;
        int particle_count;
        Data best_x;
        double best_y;

        double inertia_max;
        double inertia_min;
        double inertia;
        double constrictor_factor;
        double ff;
        double c1,c2,c3;
        mt19937 eng;


        void step();
        bool terminated();
    public:
        APSO(Problem *p,int num_particles,int max_iters);
        ~APSO();

        void solve();

        // ESE procedure
        double eucleidian_distance(Data &particle_1,Data &particle_2);
        double particle_mean_distance(int particle_idx);
        double evaluationary_factor();
        void ESE();

        Data get_best_x()const;
        double get_best_y()const;
};