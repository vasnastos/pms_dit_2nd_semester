#include "problem.hpp"


Problem::Problem() {}

Problem::Problem(int d):dimension(d) {
    this->eng=mt19937(high_resolution_clock::now().time_since_epoch().count());
    this->left_bound.resize(d);
    this->right_bound.resize(d);
}

void Problem::set_dimension(int dim) {
    this->dimension=dim;
}

void Problem::set_margins(const double &left_margin,const double &right_margin)
{
    this->margins.param(std::uniform_real_distribution<double>::param_type(left_margin,right_margin));
}

void Problem::set_left_bound(const double &value)
{
    fill(this->left_bound.begin(),this->left_bound.end(),value);
}

void Problem::set_right_bound(const double &value)
{
    fill(this->right_bound.begin(),this->right_bound.end(),value);
}
void Problem::set_left_bound(const Data &data)
{
    this->left_bound=data;
}

void Problem::set_right_bound(const Data &data)
{
    this->right_bound=data;
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

bool Problem::is_point_in(Data &x)
{
    for(int i=0,t=x.size();i<t;i++)
    {
        if(x[i]<this->left_bound[i] || x[i]>this->right_bound[i])
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
