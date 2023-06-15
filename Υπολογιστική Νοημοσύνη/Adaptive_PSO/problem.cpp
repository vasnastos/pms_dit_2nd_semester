#include "problem.hpp"


Problem::Problem() {}

Problem::Problem(int d):dimension(d) {
    this->eng=mt19937(high_resolution_clock::now().time_since_epoch().count());
}

void Problem::set_dimension(int dim) {
    this->dimension=dim;
    this->set_neural_left_margin(-10);
    this->set_neural_right_margin(10);
}

void Problem::set_margins(const double &left_margin,const double &right_margin)
{
    this->margins.param(std::uniform_real_distribution<double>::param_type(left_margin,right_margin));
}

void Problem::set_neural_left_margin(const double &value)
{
    this->neural_network_left_margin=value;
}

void Problem::set_neural_right_margin(const double &value)
{
    this->neural_network_right_margin=value;
}

int Problem::get_dimension()const
{
    return this->dimension;
}


Data Problem::get_sample()
{
    Data coefficients;
    coefficients.resize(this->dimension);
    for(int i=0;i<this->dimension;i++)
    {
        coefficients[i]=this->margins(this->eng);
    }
    return coefficients;
}

double Problem::get_left_margin()const
{
    return this->margins.a();
}

double Problem::get_right_margin()const
{
    return this->margins.b();
}

double Problem::get_neural_left_margin()const
{
    return this->neural_network_left_margin;
}

double Problem::get_neural_right_margin()const
{
    return this->neural_network_right_margin;
}

bool Problem::is_point_in(Data &x)
{
    for(int i=0,t=x.size();i<t;i++)
    {
        if(x[i]<this->neural_network_left_margin || x[i]>this->neural_network_right_margin)
        {
            return false;
        }
    }
    return true;
}

Problem::~Problem() {}

double Problem::grms(Data &x)
{
    Data gradients=this->gradient(x);
    double s=0.0;
    for(int i=0,t=x.size();i<t;i++)
    {
        s+=pow(gradients[i],2);
    }
    return sqrt(s/x.size());
}
