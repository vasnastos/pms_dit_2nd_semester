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

        void step();
        bool terminated();

        mt19937 eng;

    public:
        APSO(Problem *p,int num_particles,int max_iters);
        ~APSO();

        void solve();

        Data get_best_x()const;
        double get_best_y()const;
};