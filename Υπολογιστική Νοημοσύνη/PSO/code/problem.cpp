#include "problem.hpp"

Problem::Problem(int d):dimension(d) {
    this->eng=mt19937(random_device{});
    this->left_margin.resize(d);
    this->right_margin.resize(d);
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
    uniform_real_distribution <double> rand_real(0,1);

    Data coefficients;
    coefficients.resize(this->dimension);
    for(int i=0;i<this->dimension;i++)
    {
        coefficients[i]=this->left_margin[i]+(this->right_margin[i]-this->left_margin[i])*rand_real(this->eng);
    }
    return coefficients;
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