#include "bfgs.hpp"

double BFGS::norm_grad()
{
    Data grad=this->problem->gradient(this->xpoint);
    double norm_grad=0.0;
    for(size_t i=0,t=this->xpoint.size();i<t;i++)
    {
        norm_grad+=pow(grad[i],2);
    }
    return sqrt(norm_grad);
}

void BFGS::step()
{
   this->gradients=this->problem->gradient(this->xpoint);
   Data p(this->xpoint.size(),0.0);
   for(int i=0,ts=this->xpoint.size();i<ts;i++)
   {
        p[i]=-this->Hessian[i*ts+i]*gradients[i];
   }

   Data xnew;
   double ynew;
   double t=1.0;
   bool start=true;
   // LineSearch
   do{
        xnew=this->xpoint;
        for(int i=0,ts=this->xpoint.size();i<t;i++)
        {
            xnew[i]=xnew[i]+t*p[i];
        }
        ynew=this->problem->statFunmin(xnew);
        if(start)
        {
            start=false;
        }
        else{
            t*=this->beta;
        }
  }while(ynew-this->ypoint-alpha*t*this->norm_grad()>=0);


    Data newgradients;
    for(int i=0,ts=this->xpoint.size();i<t;i++)
    {
        this->xpoint[i]=this->xpoint[i]+t*p[i];
    }
    newgradients=this->problem->gradient(this->xpoint);
    Data y(this->xpoint.size(),0.0);
    for(size_t i=0,ts=this->xpoint.size();i<t;i++)
    {
        y[i]=newgradients[i]-this->gradients[i];
    }


    Data s(this->xpoint.size(),0.0);
    for(size_t i=0,ts=this->xpoint.size();i<t;i++)
    {
        s[i]=t*p[i];
    }

    Data Hs(this->xpoint.size(),0.0);
    for(size_t i=0,ts=this->xpoint.size();i<t;i++)
    {
        for(size_t j=0;j<ts;j++)
        {
            Hs[i]=this->Hessian[i*this->xpoint.size()+j]*s[j];
        }
    }

    double ys=0;
    for(size_t i=0,ts=this->xpoint.size();i<ts;i++)
    {
        ys+=y[i]*s[i];
    }

    for(size_t i=0,ts=this->xpoint.size();i<t;i++)
    {
        for(int j=0;j<ts;j++)
        {
            // this->Hessian[i*ts+j]=(ys+y[i]*Hs[j])/pow(this->norm_grad(),2.0)-(Hs[i]*Hs[j])/ys;
            this->Hessian[i * ts + j] += (ys + y[i] * y[j]) / ys - (Hs[i] * Hs[j]) / ys;
        }
    }
    this->ypoint=this->problem->statFunmin(this->xpoint);
    cout<<"BFGS| Iter:"<<this->iter_id<<"\tObjective:"<<this->ypoint<<"\tGRNORM:"<<this->norm_grad()<<endl;
    this->iter_id++;
}

bool BFGS::termination()
{
    return this->iter_id>this->max_iters || this->norm_grad()<1e-5;
}

BFGS::BFGS(Problem *in_problem,Data &initial_guess,int max_iters):problem(in_problem),max_iters(max_iters),iter_id(1)
{
    this->xpoint=initial_guess;
    this->ypoint=this->problem->statFunmin(this->xpoint);
    this->alpha=0.1;
    this->beta=0.5;
    this->tolerance=1e-5;
}


void BFGS::solve()
{
    this->Hessian.resize(this->xpoint.size()*this->xpoint.size());
    for(int i=0,ts=this->xpoint.size();i<ts;i++)
    {
        for(int j=0;j<ts;j++)
        {
            if(i==j)
            this->Hessian[i*ts+j]=1;
            else 
            this->Hessian[i*ts+j]=0;
        }
    }

    do
    {
        this->step();
    }while(!this->termination());
}

Data BFGS::get_best_x()
{
    return this->xpoint;
}