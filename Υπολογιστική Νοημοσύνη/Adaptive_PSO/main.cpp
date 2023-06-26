#include "apso.hpp"

class RosenBrock:public Problem{
    public:
        RosenBrock(int n):Problem(n) {
            this->set_margins(-10,10);
            this->set_left_bound(-10);
            this->set_right_bound(10);
        }
        
        double minimize_function(Data &x)
        {
            double acc_value=0.0;
            for(int i=0;i<this->dimension-1;i++)
            {
                acc_value+=100.0*pow(x[i+1]-pow(x[i],2),2)+(pow(1-x[i],2));
            }
            return acc_value;
        }

        Data gradient(Data &x)
        {
            Data gradient_points;
            gradient_points.resize(this->dimension);

            for(int i=0;i<this->dimension-1;i++)
            {
                gradient_points[i]=-400.0*(x[i+1]-pow(x[i],2))*x[i]-2.0;
            }
            gradient_points[this->dimension-1]=200.0*(x[this->dimension-1]-pow(x[this->dimension-2],2));
            return gradient_points;
        }

};

class Schwefel:public Problem
{
    public:
        Schwefel(int n):Problem(n)
        {
            this->set_margins(-500,500);
            this->set_left_bound(-500);
            this->set_right_bound(500);
        }

        double minimize_function(Data &x)
        {
            double s=0;
            for(auto &xi:x)
            {
                s+=-xi*sin(sqrt(fabs(xi)));
            }
            return s;
        }

        Data gradient(Data &x)
        {
            Data gradient_points;
            gradient_points.resize(x.size());
            for(int i=0,t=x.size();i<t;i++)
            {
                gradient_points[i]=sin(sqrt(x[i]))+(x[i]/2)*cos(sqrt(x[i])) * pow(x[i],-0.5);
            }

            return gradient_points;
        }
};


int main(int argc,char **argv)
{
    RosenBrock problem(8);

    APSO solver(&problem,100,20000);
    solver.solve();

    return EXIT_SUCCESS;
}