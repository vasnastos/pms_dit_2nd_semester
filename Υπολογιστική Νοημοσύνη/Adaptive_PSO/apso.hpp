#include "collection.hpp"

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


        void step();
        void terminated();

    public:
        APSO(Problem *p,int num_particles,int max_iters);
        ~APSO();

        void solve();

        Data get_best_x()const;
        double get_best_y()const;
};