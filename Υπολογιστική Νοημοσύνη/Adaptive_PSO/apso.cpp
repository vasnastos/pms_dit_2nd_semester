#include "apso.hpp"


APSO::APSO(Problem *p,int num_particles,int number_of_max_iters):problem(p),iter_id(0),particle_count(num_particles),max_iters(number_of_max_iters)
{
    this->c1=0.72984;
    this->c2=0.72984;
    this->c3=0.72984;

    this->best_x=Data();
    this->best_y=static_cast<double>(INT_MAX);

    this->eng=mt19937(high_resolution_clock::now().time_since_epoch().count());
}

APSO::~APSO() {}

void APSO::step()
{
    this->iter_id++;
    Data x,velocity_x,best_positional_x;
    double y,velocity_y,best_positional_y;

    double r1,r2;
    int pdimension=this->problem->get_dimension();
    uniform_real_distribution <double> rand_real_eng(0,1);

    this->inertia=this->inertia_max-(this->inertia_max-this->inertia_min)*(this->iter_id/this->max_iters);
    velocity_y=0;
    for(int i=0;i<this->particle_count;i++)
    {
        x.clear();
        velocity_x.clear();
        best_positional_x.clear();

        this->particles.get_point(i,x,y);
        this->velocity.get_point(i,velocity_x,velocity_y);
        this->best_particle.get_point(i,best_positional_x,best_positional_y);

        for(int j=0;j<pdimension;j++)
        {
            r1=rand_real_eng(this->eng);
            r2=rand_real_eng(this->eng);

            velocity_x[j]=this->inertia * velocity_x[j] + this->c1 * r1 * (best_positional_x[j]-x[j])+this->c2 * r2 * (this->best_x[j] - x[j]);
        }
        this->velocity.replace_point(i,velocity_x,velocity_y);

        for(int j=0;j<pdimension;j++)
        {
            x[j]=x[j]+velocity_x[j];
        }

        if(!this->problem->is_point_in(x)) continue;
        y=this->problem->minimize_function(x);

        this->particles.replace_point(i,x,y);
        if(y<best_positional_y)
        {
            this->best_x=x;
            this->best_y=y;
        }
    }
    cout<<"ITER:"<<this->iter_id<<"\tError(Objective):"<<this->best_y<<"%"<<endl;
}

bool APSO::terminated()
{
    double miny,maxy;
    this->particles.get_best_worst_values(miny,maxy);
    return this->iter_id>this->max_iters || fabs(maxy-miny)<=1e-4;
}

void APSO::solve()
{
    this->iter_id=0;
    uniform_real_distribution <double> rand_real(0,1);
    Data x,velocity_x;
    double y,velocity_y=0;

    this->ff=this->c1+this->c2;
    this->constrictor_factor=2.0/(2.0-this->ff-sqrt(pow(this->ff,2)-4.0*this->ff));

    for(int i=0;i<this->particle_count;i++)
    {
        x=this->problem->get_sample();
        y=this->problem->minimize_function(x);

        this->particles.add_point(x,y);
        this->best_particle.add_point(x,y);

        if(i==0  || y<this->best_y)
        {
            this->best_x=x;
            this->best_y=y;
        }
        velocity_x.resize(x.size());
        for(int j=0,t=velocity_x.size();j<t;j++)
        {
            velocity_x[j]=rand_real(this->eng);
        }
        this->velocity.add_point(velocity_x,velocity_y);
    }

    do{
        this->step();
    }while(!this->terminated());
    cout<<endl<<endl;
}

Data APSO::get_best_x()const
{
    return this->best_x;
}

double APSO::get_best_y()const
{
    return this->best_y;
}