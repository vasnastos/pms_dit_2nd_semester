#include "pso.hpp"

mt19937 gen(high_resolution_clock::now().time_since_epoch().count());

Data Particle::left_bound=Data();
Data Particle::right_bound=Data();

Particle::Particle() {}

Particle::Particle(Data &x)
{
    uniform_real_distribution <double> rande(0,1);
    this->position=x;
    this->velocity.resize(this->position.size());
    for(int i=0,t=this->velocity.size();i<t;i++)
    {
        this->velocity[i]=rande(gen);
    }

    this->best_position=this->position;
    this->best_objective=this->objective_value;
}

Particle::Particle(Data &x,double &obj_value)
{
    uniform_real_distribution <double> rande(0,1);
    this->position=x;
    this->velocity.resize(x.size());
    for(int i=0,t=this->velocity.size();i<t;i++)
    {
        this->velocity[i]=rande(gen);
    }
    this->objective_value=obj_value;

    this->best_position=this->position;
    this->best_objective=obj_value;
}


void Particle::update_velocity(double &inertia,double &c1,double &c2,Particle &global_best_particle)
{
    double r1,r2;
    uniform_real_distribution <double> rande(0,1);
    for(int i=0,t=this->velocity.size();i<t;i++)
    {
        r1=rande(gen);
        r2=rande(gen);
        this->velocity[i]=inertia*this->velocity[i]+c1*r1*(this->best_position[i]-this->position[i])+c2*r2*(global_best_particle.position[i]-this->position[i]);
    }
}

bool Particle::update_position()
{
    Data temp=this->position;
    for(int i=0,t=this->position.size();i<t;i++)
    {
        temp[i]=this->position[i]+this->velocity[i];
        if(temp[i]<Particle::left_bound[i] || temp[i]>Particle::right_bound[i])
        {
            return false;
        }
    }

    // Similarity check
    double similarity_check=0.0;
    for(int i=0,t=this->position.size();i<t;i++)
    {
        similarity_check=fabs(temp[i]-this->position[i]);
    }

    if(similarity_check<=1e-5)
    {
        return false;
    }

    for(int i=0,t=this->position.size();i<t;i++)
    {
        this->position[i]=this->position[i]+this->velocity[i];
    }
    return true;
}

void Particle::update_objective_value(double &new_obj_value,Particle &global_best_particle)
{
    this->objective_value=new_obj_value;
    if(new_obj_value<this->best_objective)
    {
        this->best_position=this->position;
        this->best_objective=new_obj_value;
    }

    if(new_obj_value<global_best_particle.objective_value)
    {
        global_best_particle=*this;
    }
}

PSO::PSO(Problem *data,int num_particles,int num_iterations):problem(data),num_particles(num_particles),max_iters(num_iterations)
{
    Particle::left_bound=data->getLeftMargin();
    Particle::right_bound=data->getRightMargin();

    this->inertia_max=0.9;
    this->inertia_min=0.4;

    // this->c1=2.0;
    // this->c2=2.0;

    this->c1=1.49618;
    this->c2=1.49618;

    this->iter=0;

    uniform_int_distribution <int> rand_int(1,5);
    this->k=rand_int(gen);
    this->midpoint=this->max_iters/2.0;

    this->v1=0;
    this->v2=0;
    this->variance=0;

}

void PSO::update_inertia()
{
    // this->inertia=this->inertia_max-(this->inertia_max-this->inertia_min)*(static_cast<double>(this->iter)/static_cast<double>(this->max_iters));
    this->inertia=this->inertia_min+(this->inertia_max-this->inertia_min)/(1+exp(-k*(this->iter-this->midpoint))); 
    
    // Expontential update rule
    // this->inertia=this->inertia*exp(-1e-4*this->iter);
}

void PSO::step()
{
    bool updated;
    double objective;

    this->iter++;
    this->update_inertia();

    for(int i=0,t=this->particles.size();i<t;i++)
    {
        this->particles.at(i).update_velocity(this->inertia,this->c1,this->c2,this->best_particle);
        updated=this->particles.at(i).update_position();
        if(!updated) continue;
        objective=this->problem->statFunmin(this->particles.at(i).position);
        this->particles.at(i).update_objective_value(objective,this->best_particle);
    }
    cout<<"PSO| Iter:"<<this->iter<<"\tObjective:"<<this->best_particle.objective_value;
    this->objectives.emplace_back(this->best_particle.objective_value);
}

bool PSO::terminated()
{
    double max_obj,min_obj;
    this->get_best_worst_objective(max_obj,min_obj);
    cout<<"\tMin Obj:"<<min_obj<<"\tMax Obj:"<<max_obj<<endl;
    
    this->v1+=fabs(this->best_particle.objective_value);
    this->v2+=pow(this->best_particle.objective_value,2);
    double v=(this->v2/static_cast<double>(this->iter)) - pow(this->v1/static_cast<double>(this->iter),2);

    if(fabs(this->best_particle.objective_value-this->old_best_particle.objective_value)>1e-5)
    {
        this->variance=v/2.0;
        this->old_best_particle=this->best_particle;
    }

    return this->iter>=this->max_iters || fabs(max_obj-min_obj)<=1e-4 || fabs(min_obj-0.0)<=1e-4;
}

void PSO::get_best_worst_objective(double &max_obj,double &min_obj)
{
    max_obj=this->particles.at(0).objective_value;
    min_obj=this->particles.at(0).objective_value;

    for(int i=1;i<this->num_particles;i++)
    {
        if(this->particles.at(i).objective_value>max_obj)
        {
            max_obj=this->particles.at(i).objective_value;
        }
        else if(this->particles.at(i).objective_value<min_obj)
        {
            min_obj=this->particles.at(i).objective_value;
        }
    }
}

void PSO::solve()
{
    Data new_sample;
    double objective;
    this->particles.clear();
    this->best_particle.position=Data();
    this->best_particle.objective_value=1e+100;

    for(int i=0;i<this->num_particles;i++)
    {
        new_sample=this->problem->getSample();
        objective=this->problem->statFunmin(new_sample);  
        this->particles.emplace_back(Particle(new_sample,objective));
        if(objective<this->best_particle.objective_value)
        {
            this->best_particle=this->particles.at(i);
        }
    }

    this->old_best_particle=this->best_particle;

    do
    {
        this->step();
    } while (!this->terminated());
    
}

Data PSO::get_best_x()
{
    return this->best_particle.position;
}

void PSO::save(string filename)
{
    fs::path dpath;
    for(auto &x:{"..","results","distribution"})
    {
        dpath.append(x);
    }
    dpath.append(filename);

    fstream fp;
    fp.open(dpath.string(),ios::out);

    for(auto &yvalue:this->objectives)
    {
        fp<<yvalue<<endl;
    }
    fp.close();
}