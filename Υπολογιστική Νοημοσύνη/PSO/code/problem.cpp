#include "problem.hpp"


Problem::Problem() {}

Problem::Problem(int d):dimension(d) {
    this->eng=mt19937(high_resolution_clock::now().time_since_epoch().count());
}

void Problem::set_dimension(int dim) {
    this->dimension=dim;
}

void Problem::set_margins(double &left_margin,double &right_margin)
{
    this->margins.param(uniform_real_distribution<double>::param_type(left_margin,right_margin));
}

pair <double,double> Problem::get_margins()const
{
    return make_pair(this->margins.a(),this->margins.b());
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

