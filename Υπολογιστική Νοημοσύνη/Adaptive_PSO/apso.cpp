#include "apso.hpp"


APSOhyperparameters::APSOhyperparameters(double in_max,double in_min,double c1_param,double c2_param,double cf,double ff_param,double elrmn,double elrmx):inertia_max(in_max),inertia_min(in_min),c1(c1_param),c2(c2_param),ff(ff_param),elitist_learning_rate_min(elrmn),elitist_learning_rate_max(elrmx) {}
APSOhyperparameters::APSOhyperparameters()
{
    this->inertia_max=0.9;
    this->inertia_min=0.4;

    this->c1=2.0;
    this->c2=2.0;

    this->ff=this->c1+this->c2;
    this->constrictor_factor=2.0/fabs(2-this->ff-sqrt(pow(this->ff,2)-4*this->ff));

    this->elitist_learning_rate_min=0.1;
    this->elitist_learning_rate_max=1.0;
}


APSO::APSO(Problem *p,int num_particles,int number_of_max_iters):problem(p),iter_id(0),particle_count(num_particles),max_iters(number_of_max_iters)
{
    this->best_x=Data();
    this->best_y=numeric_limits<double>::infinity();

    this->previous_state=0;
    this->adaptive_set=new FuzzySet(5);
    this->adaptive_set->set_membership(0,INT_MIN);
    map <int,int> fuzzyset_rulebase{
        {0,1},
        {1,2},
        {2,3},
        {3,4},
        {4,1}
    };
    this->adaptive_set->set_rule_base(fuzzyset_rulebase);
    this->eng=mt19937(high_resolution_clock::now().time_since_epoch().count());
}

APSO::~APSO() {delete this->adaptive_set;}

void APSO::step()
{
    this->iter_id++;
    
    this->ESE();
    // this->inertia=this->params.inertia_max-(this->params.inertia_max-this->params.inertia_min)*(static_cast<double>(this->iter_id)/static_cast<double>(this->max_iters));
    Data x,velocity_x,best_positional_x;
    double y,velocity_y,best_positional_y;
    double r1,r2;
    int pdimension=this->problem->get_dimension();
    uniform_real_distribution <double> rand_real_eng(0,1);
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

            velocity_x[j]= this->params.constrictor_factor*(this->inertia * velocity_x[j] + this->params.c1 * r1 * (best_positional_x[j]-x[j])+this->params.c2 * r2 * (this->best_x[j] - x[j]));
            // velocity_x[j]=this->params.constrictor_factor *(velocity_x[j] + this->params.c1 * r1 * (best_positional_x[j]-x[j])+this->params.c2 * r2 * (this->best_x[j]-x[j]))
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
            this->best_particle.replace_point(i,best_positional_x,best_positional_y);
        }

        if(y<this->best_y)
        {
            this->global_best_particle_idx=i;
            this->best_x=x;
            this->best_y=y;
        }
    }
    this->print_global_best();
}

void APSO::print_global_best()
{
    cout<<"ITER:"<<this->iter_id;
    cout<<"  X[";
    for(auto &xpoint:this->best_x)
    {
        cout<<xpoint<<" ";
    }
    cout<<"]\tc1:"<<this->params.c1<<"\tc2:"<<this->params.c2<<"\tBest Objective:"<<this->best_y;
}

bool APSO::terminated()
{
    double miny,maxy;
    this->particles.get_best_worst_values(miny,maxy);

    cout<<"\tMax:"<<maxy<<"\tMin:"<<miny<<endl;

    return this->iter_id>this->max_iters || fabs(maxy-miny)<=1e-4;
}

void APSO::solve()
{
    this->iter_id=0;
    uniform_real_distribution <double> rand_real(0,1);
    Data x,velocity_x;
    double y,velocity_y=0;

    for(int i=0;i<this->particle_count;i++)
    {
        x=this->problem->get_sample();
        y=this->problem->minimize_function(x);

        this->particles.add_point(x,y);
        this->best_particle.add_point(x,y);

        if(i==0  || y<this->best_y)
        {
            this->global_best_particle_idx=i;
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


// ESE procedure
double APSO::eucleidian_distance(Data &particle_1,Data &particle_2)
{
    double s=0;
    for(int i=0,t=this->problem->get_dimension();i<t;i++)
    {
        s+=pow(particle_1.at(i)-particle_2.at(i),2);
    }
    return sqrt(s);
}

double APSO::particle_mean_distance(int particle_idx)
{
    double s=0;
    Data particle_xi,particle_xj;
    double particle_yi,particle_yj;
    this->particles.get_point(particle_idx,particle_xi,particle_yi);
    for(int particle_jdx=0;particle_jdx<this->particle_count;particle_jdx++)
    {
        if(particle_idx==particle_jdx) continue;
        this->particles.get_point(particle_jdx,particle_xj,particle_yj);
        s+=this->eucleidian_distance(particle_xi,particle_xj);
    }
    return s/static_cast<double>(this->particle_count-1);
}

double APSO::evaluationary_factor()
{
    Data particle_mean_distance;
    particle_mean_distance.resize(this->particle_count);

    for(int particle_idx=0;particle_idx<this->particle_count;particle_idx++)
    {
        particle_mean_distance[particle_idx]=this->particle_mean_distance(particle_idx);
    }

    double dg=particle_mean_distance.at(this->global_best_particle_idx);
    double dmin=*std::min_element(particle_mean_distance.begin(),particle_mean_distance.end());
    double dmax=*std::max_element(particle_mean_distance.begin(),particle_mean_distance.end());
    return (dg-dmin)/(dmax-dmin);

}


void APSO::adjust_acceleration_coefficients(double &acceleration_rate,string c1_direction,string c2_direction)
{
    uniform_real_distribution <double> dis(0.05,acceleration_rate);
    double c1_new,c2_new,sum;

    if(c1_direction=="increase")
    {
        c1_new=this->params.c1+dis(this->eng);
    }
    else if(c1_direction=="decrease")
    {
        c1_new=this->params.c1-dis(this->eng);
    }

    if(c2_direction=="increase")
    {
        c2_new=this->params.c2+dis(this->eng);
    }
    else if(c2_direction=="decrease")
    {
        c2_new=this->params.c2-dis(this->eng);
    }
    
    c1_new=std::clamp(c1_new,1.5,2.5);
    c2_new=std::clamp(c2_new,1.5,2.5);
    sum=c1_new+c2_new;


    if(sum>4.0)
    {
        c1_new=(c1_new/sum)*4.0;
        c2_new=(c2_new/sum)*4.0;
    }

    this->params.c1=c1_new;
    this->params.c2=c2_new;
}

void APSO::ESE()  //evolutionary state optimization
{
    double efactor=this->evaluationary_factor();
    double s1_factor=0.0;
    double s2_factor=0.0;
    double s3_factor=0.0;
    double s4_factor=0.0;

    string c1_direction;
    string c2_direction;

    bool els;

    // calculate s1_factor
    if(efactor<=0.4)
    {
        s1_factor=0.0;
    }
    else if(efactor<=0.6)
    {
        s1_factor=5*efactor-2;
    }
    else if(efactor<=0.7)
    {
        s1_factor=1;
    }
    else if(efactor<=0.8)
    {
        s1_factor=-10*efactor+8;
    }
    else if(efactor<=1)
    {
        s1_factor=0;
    }

    // calculate s2_factor
    if(efactor<=0.2)
    {   
        s2_factor=0;
    }
    else if(efactor<=0.3)
    {
        s2_factor=10*efactor-2;
    }
    else if(efactor<=0.4)
    {
        s2_factor=1.0;
    }
    else if(efactor<=0.6)
    {
        s2_factor=-5*efactor+3;
    }
    else if(efactor<=1)
    {
        s2_factor=0.0;
    }

    // calculate s3_factor
    if(efactor<=0.1)
    {
        s3_factor=1.0;
    }
    else if(efactor<=0.3)
    {
        s2_factor=-5*efactor+1.5;
    }
    else if(efactor<=1)
    {
        s3_factor=0.0;
    }

    // calculate s4_factor
    if(efactor<=0.7)
    {
        s4_factor=0.0;
    }
    else if(efactor<=0.9)
    {
        s4_factor=5.0*efactor-3.5;
    }
    else if(efactor<=1)
    {
        s4_factor=1.0;
    }
    
    //set fuzzy set
    this->adaptive_set->set_membership(1,s1_factor);
    this->adaptive_set->set_membership(2,s2_factor);
    this->adaptive_set->set_membership(3,s3_factor);
    this->adaptive_set->set_membership(4,s4_factor); 

    //Update State
    int current_state=this->adaptive_set->fuzzy_decision(this->previous_state);
    this->previous_state=current_state;

    double acceleration_rate;
    switch(current_state)
    {
        case 1:
            acceleration_rate=0.1;
            c1_direction="increase";
            c2_direction="decrease";
            els=true;
            break;
        case 2:
            acceleration_rate=0.5;
            c1_direction="increase";
            c2_direction="decrease";
            els=false;
            break;
        case 3:
            acceleration_rate=0.5;
            c1_direction="increase";
            c2_direction="increase";
            els=false;
            break;
        case 4:
            acceleration_rate=0.1;
            c1_direction="decrease";
            c2_direction="increase";
            els=false;
            break;
        default: 
            els=false;
            break;
    }
    this->adjust_acceleration_coefficients(acceleration_rate,c1_direction,c2_direction);
    if(els)
    {
        this->ELS();
    }

    // Adaptive Control of the hyperparameters update inertia and accelaration coefficients
    // this->inertia=1.0/(1.0*exp(-2.6*efactor));
    this->inertia=this->params.inertia_max-(this->params.inertia_max-this->params.inertia_min)*(static_cast<double>(this->iter_id)/static_cast<double>(this->max_iters));
}

void APSO::ELS()
{
    Data P=this->best_x;
    uniform_real_distribution <double> dim_selector(0,this->problem->get_dimension()-1);
    int dimension_selected=dim_selector(this->eng);
    this->elitist_learning_rate=this->params.elitist_learning_rate_max-(this->params.elitist_learning_rate_max-this->params.elitist_learning_rate_min)*(static_cast<double>(this->iter_id)/static_cast<double>(this->max_iters));
    normal_distribution <double> gaussian(0.0,this->elitist_learning_rate);
    P[dimension_selected]=P.at(dimension_selected)+(this->problem->get_right_margin()-this->problem->get_left_margin())*gaussian(this->eng);
    if(this->problem->is_point_in(P))
    {
        double objective_value=this->problem->minimize_function(P);
        if(objective_value<this->best_y)
        {
            this->best_x=P;
            this->best_y=objective_value;
        } 
        else
        {
            size_t worst_objective_idx=this->particles.get_worst_objective_collection_idx();
            this->particles.replace_point(worst_objective_idx,P,objective_value);
        }
    }

}

Data APSO::get_best_x()const
{
    return this->best_x;
}

double APSO::get_best_y()const
{
    return this->best_y;
}