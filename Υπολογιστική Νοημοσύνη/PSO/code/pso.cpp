#include "pso.hpp"

PSO::PSO(Problem *p,int number_of_iterations=1000,int particle_count=500):problem(p),max_iters(number_of_iterations),particle_count(particle_count),iter(0),inertia_max(0.999),inertia_min(0.4) {
    this->eng=mt19937(random_device{});
    
    // // Standard PSO
    // this->c1=2.0;
    // this->c2=2.0;
    // this->c3=2.0;

    // // Constriction factor PSO
    // this->c1=1.49618;
    // this->c2=1.49618;
    // this->c3=1.49618;

    // Clerc's PSO(emphasis on global best position)
    this->c1=0.72984;
    this->c2=0.72984;
    this->c3=0.72984;
}

PSO::~PSO() {}

void PSO::set_max_iters(int number_of_iterations)
{
    this->max_iters=number_of_iterations;
}

int PSO::get_max_iters()const
{
    return this->max_iters;
}

void PSO::set_particle_count(int cnt)
{
    this->particle_count=cnt;
}

int PSO::get_particle_count()const
{
    return this->particle_count;
}

Data PSO::get_best_x()
{
    return this->best_x;
}
double PSO::get_best_y()
{
    return this->best_y;
}

bool PSO::terminated()
{
    double miny,maxy;
    this->particle.get_best_worst_values(miny,maxy);
    return this->iter>this->max_iters || fabs(maxy-miny)<1e-4;
}

void PSO::step()
{
    this->iter++;
    Data x,velocity_x,bx;
    double y,velocity_y,by;
    double r1,r2,r3;
    int problem_dimension=this->problem->get_dimension();
    uniform_real_distribution <double> rand_real_eng(0,1);
    Data geometric_center_points;

    for(int i=0;i<this->particle_count;i++)
    {   
        // geometric_center_points=this->geometric_center();
        this->particle.get_point(i,x,y);
        this->velocity.get_point(i,velocity_x,velocity_y);
        this->best_particle.get_point(i,bx,by);

        this->inertia=this->inertia_max-(this->inertia_max-this->inertia_min)*this->iter/this->max_iters;

        // update velocity
        for(int j=0;j<problem_dimension;j++)
        {
            r1=rand_real_eng(eng);
            r2=rand_real_eng(eng);
            r3=rand_real_eng(eng);

            velocity_x[j]=this->inertia*velocity_x[j]+this->c1*r1*(x[j]-bx[j])+this->c2*r2*(x[j]-bx[j]);
            // velocity_x[j]=this->inertia*velocity_x[j]+this->c1*r1*(x[j]-bx[j])+this->c2*r2*(x[j]-bx[j])+r3*c3*geometric_center_points[j];
        }
        this->velocity.replace_point(i,velocity_x,velocity_y);

        // update x point of a particle(collection)
        for(int j=0;j<problem_dimension;j++)
        {
            x[j]=x[j]*velocity_x[j];
        }

        // Check problem margins
        if(!this->problem->is_point_in(x)) continue;


        y=this->problem->minimize_function(x);
        this->particle.replace_point(i,x,y);


        // Check for best points
        if(y<by)
        {
            best_particle.replace_point(i,x,y);
        }
        if(y<this->best_y)
        {
            this->best_x=x;
            this->best_y=y;
            cout<<"ITER:"<<this->iter<<"\tObjective:"<<this->best_y<<"\tProblem:"<<this->problem->description()<<endl;
        }
    }
}

void PSO::solve()
{
    this->iter=0;
    uniform_real_distribution<double> rand_real;
    Data x,velocity_;
    double y,velocity_y=0;


    // Initialize particles and velocity
    for(int i=0;i<this->particle_count;i++)
    {
        x=this->problem->get_sample();
        y=this->problem->minimize_function(x);
        this->particle.add_point(x,y);
        this->best_particle.add_point(x,y);
        if(i==0 || y<this->best_y)
        {
            this->best_x=x;
            this->best_y=y;
        }

        velocity_.resize(x.size());
        for(int j=0,t=velocity_.size();j<t;j++)
        {
            velocity_[j]=rand_real(this->eng);
        }
        this->velocity.add_point(velocity_,velocity_y);
    }

    // Solve the problem
    do
    {
        this->step();
    } while (!this->terminated());
    cout<<endl<<endl;
}

Data PSO::geometric_center()
{
    Data geometric_center_points;
    geometric_center_points.resize(this->problem->get_dimension());
    fill(geometric_center_points.begin(),geometric_center_points.end(),0.0);


    Data xpoint;
    double ypoint;
    for(int i=0;i<this->particle_count;i++)
    {
        this->particle.get_point(i,xpoint,ypoint);
        for(int j=0,t=xpoint.size();j<t;j++)
        {
            geometric_center_points[j]+=xpoint.at(j);
        }
    }

    for(int j=0,t=this->problem->get_dimension();j<t;j++)
    {
        geometric_center_points[j]/=this->particle_count;
    }

    return geometric_center_points;
}