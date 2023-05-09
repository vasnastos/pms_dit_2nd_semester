#include "problem.h"

Problem::Problem(int d):dimension(d) {
    this->mt=mt19937(random_device{});
    this->left_margin.resize(d);
    this->right_margin.resize(d);
}

void Problem::set_weight_init_method(string winit)
{
    this->weight_initialization=winit;
}

string Problem::get_weight_init_method()const
{
    return this->weight_initialization;
}

void Problem::set_left_margin(Data &x)
{
    this->left_margin=x;
}

void Problem::set_right_margin(Data &x)
{
    this->right_margin=x;
}

int Problem::get_dimension()const
{
    return this->dimension;
}

Data Problem::get_left_margin()const
{
    return this->left_margin;
}

Data Problem::get_right_margin()const
{
    return this->right_margin;
}

Data Problem::get_sample()
{
    uniform_real_distribution <double> rand_real;
    if(this->weight_initialization=="Random")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-0.01,0.01));
    }
    else if(this->weight_initialization=="Xavier")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-(1/this->dimension),1/this->dimension));
    }
    else if(this->weight_initialization=="UXavier")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-(6/this->dimension),(6/this->dimension)));
    }

    Data weight_set;
    weight_set.resize(this->dimension);
    for(int i=0;i<this->dimension;i++)
    {
        weight_set[i]=rand_real(mt);
    }
    return weight_set;
}

bool Problem::is_point_in(Data &x)
{
    for(int i=0,t=x.size();i<t;i++)
    {
        if(x[i]<this->left_margin[i] || x[i]>this->right_margin[i])
        {
            return false;
        }
    }
    return true;
}

Problem::~Problem() {}