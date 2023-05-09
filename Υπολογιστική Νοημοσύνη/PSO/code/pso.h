#include "problem.h"
#include "collection.h"

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
        int iter;
        size_t max_iters;

        Data best_x;
        double best_y;

        int c1,c2;
        mt19937 eng;

    public:
        PSO(Problem *p,int number_of_iterations=1000,int particle_count=500);
        ~PSO();

        void set_max_iters(int number_of_iterations);
        int get_max_iters()const;

        void set_particle_count(int cnt);
        int get_particle_count()const;

        Data get_best_x();
        double get_best_y();
        double geometric_center();
        

        bool terminated();
        void step();
        void solve();
};