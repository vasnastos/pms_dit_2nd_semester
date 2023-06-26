#include "problem.hpp"

Problem::Problem(int n)
{
    dimension = n;
    left.resize(dimension);
    right.resize(dimension);
    besty = 1e+100;
    functionCalls = 0;
}

double    Problem::statFunmin(Data &x)
{
    double y = funmin(x);
    if(y<besty)
    {
        besty = y;
        bestx = x;
    }
    ++functionCalls;
    return y;
}

Data    Problem::getBestx() const
{
    return bestx;
}

double  Problem::getBesty() const
{
    return besty;
}

int     Problem::getFunctionCalls() const
{
    return functionCalls;
}

int Problem::getDimension() const
{
    return dimension;
}


/** Dimiourgei me omoiomorfi katanomi
 *  ena neo simeio sto pedio orismou tis synartisis.
 *  Sta neuronika diktya epistrefei ena neo synolo
 *  parametron
    x = a+(b-a)*r, r in[0,1]
**/
Data Problem::getSample()
{
    Data x;
    x.resize(dimension);
    double r;
    for (int i = 0; i < dimension; i++)
    {
        r = ((double)rand()/(double)RAND_MAX);
        if (r < 0)
            r = -r;
        x[i] = 2.0 * r-1.0;
        //x[i] = left[i] + (right[i] - left[i]) * r;
    }
    return x;
}
void Problem::setLeftMargin(Data &x)
{
    left = x;
}

void Problem::setRightMargin(Data &x)
{
    right = x;
}
Data Problem::getLeftMargin() const
{
    return left;
}

Data Problem::getRightMargin() const
{
    return right;
}

double Problem::grms(Data &x)
{
    Data g = gradient(x);
    int i;
    double s = 0.0;
    for (i = 0; i < x.size(); i++)
        s = s + g[i] * g[i];
    return sqrt(s / x.size());
}

Problem::~Problem()
{
}

bool Problem::isPointInside(Data &x)
{
    for(int i=0,t=x.size();i<t;i++)
    {
        if(x[i]<this->left[i] || x[i]>this->right[i])
        {
            return false;
        }
    }
    return true;
}

Category Problem::category()
{
    return Category::REG;
}
