#include "collection.hpp"

Collection::Collection() {}

Collection::~Collection() {}

void Collection::add_point(Data &x,double &y)
{
    this->xaxis_data.emplace_back(x);
    this->yaxis_data.emplace_back(y);
}

void Collection::get_point(int index,Data &x,double &y)
{
    if(index<0 || index>this->xaxis_data.size())
    {
        return;
    }

    x=this->xaxis_data.at(index);
    y=this->yaxis_data.at(index);
}

bool Collection::have_graph_minima(Data &x,double &y,double distance)
{

}

void Collection::resize_in_fraction(double fraction)
{
    double tempy;
    Data tempx;
    for(int i=0,t=this->xaxis_data.size();i<t;i++)
    {
        for(int j=0;j<t-1;j++)
        {
            if(this->yaxis_data[j+1]<this->yaxis_data[j])
            {
                tempx=this->xaxis_data[j];
                this->xaxis_data[j]=this->xaxis_data[j+1];
                this->xaxis_data[j+1]=tempx;

                tempy=this->yaxis_data[j];
                this->yaxis_data[j]=this->yaxis_data[j+1];
                this->yaxis_data[j+1]=tempy;
            }
        }
    }

    this->xaxis_data.resize(int(fraction* this->xaxis_data.size()));
    this->yaxis_data.resize(int(fraction*this->yaxis_data.size()));
}

int Collection::size()
{
    return this->xaxis_data.size();
}

double Collection::get_distance(Data &x,Data &y)
{
    double d=0;
    for(int i=0,t=x.size();i<t;i++)
    {
        d+=pow(x.at(i)-y.at(i),2);
    }
    return sqrt(d/x.size());
}

bool Collection::is_point_inside(Data &x,double &y)
{
    double d;
    for(auto &xvalue:this->xaxis_data)
    {
        if(this->get_distance(xvalue,x)<1e-4)
        {
            return true;
        }
    }
    return false;
}

void Collection::replace_point(int index,Data &x,double &y)
{
    if(index<0 || index>this->xaxis_data.size())
    {
        return;
    }
    this->xaxis_data[index]=x;
    this->yaxis_data[index]=y;
}

void Collection::get_best_worst_values(double &besty,double &worsty)
{
    besty=this->yaxis_data[0];
    worsty=this->yaxis_data[0];
    for(int i=1,t=this->xaxis_data.size();i<t;i++)
    {
        if(this->yaxis_data[i]<besty)
        {
            besty=this->yaxis_data[i];
        }
        if(this->yaxis_data[i]>worsty)
        {
            worsty=this->yaxis_data[i];
        }
    }
}