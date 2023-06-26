#include "mlp_problem.hpp"

/** edo einai i synartisi dimioyrgias **/
MlpProblem::MlpProblem(Dataset *d,int n)
    :Problem((d->dimension()+2)*n)
{
    data = d;
    int k = (d->dimension()+2)*n;
    left.resize(k);
    right.resize(k);
    /** edo bazo ta oria ton
     *  parametron tou neuronikou diktyou **/
    for(int i=0;i<k;i++)
    {
        left[i]=-100;
        right[i]=100;
    }
    weight.resize(k);
    initMethod="smallValues";
}

void    MlpProblem::setInitMethod(string m)
{
    initMethod = m;
}

string  MlpProblem::getInitMethod() const
{
    return initMethod;
}

Data    MlpProblem::getSample()
{
    Data x;
    x.resize(getDimension());
    if(initMethod==SMALLVALUES_METHOD)
    {
        double left = -0.1;
        double right  = 0.1;
        for(int i=0;i<getDimension();i++)
        {
            x[i]=left + (right - left)*rand()*1.0/RAND_MAX;
        }
    }
    else
    if(initMethod == XAVIER_METHOD)
    {
        double left = -1.0/sqrt(data->dimension());
        double right  = 1.0/sqrt(data->dimension());
        for(int i=0;i<getDimension();i++)
        {
            x[i]=left + (right - left)*rand()*1.0/RAND_MAX;

        }
    }
    else
    if(initMethod == XAVIERNORM_METHOD)
    {
        int nodes = weight.size()/(data->dimension()+2);
        double left = -1.0/sqrt(data->dimension()+nodes);
        double right  = 1.0/sqrt(data->dimension()+nodes);
        for(int i=0;i<getDimension();i++)
        {
            x[i]=left + (right - left)*rand()*1.0/RAND_MAX;

        }
    }
    return x;
}

void    MlpProblem::setWeights(Data &w)
{
    weight  =w ;
}

/** edo ypologizoume to train error pou einai
 *  kai i timi pou elaxistopoioume **/
double MlpProblem::funmin(Data &x)
{
    weight  =x ;
    return getTrainError();

}

/** edo epistrefoume tin paragogo tis
 *  synartisis funmin(x) os pros x**/
Data    MlpProblem::gradient(Data &x)
{
    Data g;
    weight = x;
    g.resize(weight.size());
    for(int i=0;i<(int)g.size();i++)
        g[i]=0.0;
    for(int i=0;i<data->count();i++)
    {
        Data xx = data->get_xpointi(i);
        Data gtemp = getDerivative(xx);
        double per=getOutput(xx)-data->get_ypointi(i);
        for(int j=0;j<(int)g.size();j++)	g[j]+=gtemp[j]*per;
    }
    for(int j=0;j<(int)x.size();j++) g[j]*=2.0;
    return g;
}

double  MlpProblem::sig(double x)
{
    return 1.0/(1.0+exp(-x));
}

double  MlpProblem::sigder(double x)
{
    double s = sig(x);
    return s*(1.0-s);
}

/** einai i exodos tou neuronikou gia to protypo x**/
double  MlpProblem::getOutput(Data  &x)
{
    double arg=0.0;
    double per=0.0;
    int nodes = weight.size()/(data->dimension()+2);
     int d = data->dimension();
    for(int i=1;i<=nodes;i++)
    {
        arg=0.0;
        for(int j=1;j<=d;j++)
        {
            int pos=(d+2)*i-(d+1)+j-1;
            arg+=weight[pos]*x[j-1];
        }
        arg+=weight[(d+2)*i-1];
        per+=weight[(d+2)*i-(d+1)-1]*sig(arg);
    }
    return per;
}

/** einai i paragogos tou neuronikou os
 *  pros to protypo x**/
Data    MlpProblem::getDerivative(Data &x)
{
    double arg;
        double f,f2;
        int nodes = weight.size()/(data->dimension()+2);
        int d = data->dimension();
        Data G;
        G.resize(weight.size());

        for(int i=1;i<=nodes;i++)
        {
                arg = 0.0;
                for(int j=1;j<=d;j++)
                {
                        arg+=weight[(d+2)*i-(d+1)+j-1]*x[j-1];
                }
                arg+=weight[(d+2)*i-1];
                f=sig(arg);
                f2=f*(1.0-f);
                G[(d+2)*i-1]=weight[(d+2)*i-(d+1)-1]*f2;
                G[(d+2)*i-(d+1)-1]=f;
                for(int k=1;k<=d;k++)
                {
                        G[(d+2)*i-(d+1)+k-1]=
                                x[k-1]*f2*weight[(d+2)*i-(d+1)-1];
                }
    }
        return G;
}

/** edo exoume to sfalma ekpaideysis **/

double  MlpProblem::getTrainError()
{
    double error = 0.0;
    if(this->data->get_category()==Category::REG)
    {
        for(int i=0;i<data->count();i++)
        {
            Data xx = data->get_xpointi(i);
            double yy = data->get_ypointi(i);
            double per = getOutput(xx);
            error+= (per-yy)*(per-yy);
        }
    }
    else
    {
        Data xi_point;
        double y_true;
        double y_pred;
        for(int i=0,t=this->data->count();i<t;i++)
        {
            xi_point=this->data->get_xpointi(i);
            y_true=this->data->get_class(i);
            y_pred=this->getOutput(xi_point);
            error+=fabs(this->data->get_class(y_pred)-y_true)>1e-4;
        }
        error=(error*100.0)/static_cast<double>(this->data->count());
    }
    return error;
}

/** kanei oti kai i getTrainError() alla gia to test set **/
double  MlpProblem::getTestError(Dataset *test)
{
    double error = 0.0;
    Data xi_point;
    double y_true;
    double y_pred;
    if(test->get_category()==Category::REG)
    {
        for(int i=0;i<test->count();i++)
        {
            xi_point = test->get_xpointi(i);
            y_true = test->get_ypointi(i);
            y_pred = getOutput(xi_point);
            error+= pow(y_pred-y_true,2);
        }
    }
    else
    {
        for(int i=0,t=test->count();i<t;i++)
        {
            xi_point=test->get_xpointi(i);
            y_true=test->get_class(i);
            y_pred=this->getOutput(xi_point);
            error+=fabs(test->get_class(y_pred)-y_true)>1e-4;
        }
        error=(error*100)/test->count();
    }
    return error;
}

/** edo epistrefo to classification sfalma sto test set **/
double  MlpProblem::getClassTestError(Dataset *test)
{
    double error = 0.0;
    for(int i=0;i<test->count();i++)
    {
        Data xx = test->get_xpointi(i);
        double realClass = test->get_class(i);
        double per = getOutput(xx);
        double estClass = test->get_class(per);
        error+= (fabs(estClass - realClass)>1e-5);
    }
    /** to metatrepoume se pososto **/
    return error*100.0/test->count();
}

MlpProblem::~MlpProblem()
{
}

void MlpProblem::optimize(string optimizer)
{
    Data new_weight_set;
    if(optimizer=="Adam")
    {
        Adam model(this);
        model.solve();
        model.save(this->save_distribution_path);
        new_weight_set=model.get_best_x();
    }
    else if(optimizer=="PSO")
    {
        PSO model(this,200,5000);
        model.solve();
        cout<<this->save_distribution_path<<endl;
        model.save(this->save_distribution_path);
        new_weight_set=model.get_best_x();
    }

    // Local search
    BFGS loptimizer(this,new_weight_set,5000);
    loptimizer.solve();
    new_weight_set=loptimizer.get_best_x();

    this->setWeights(new_weight_set);    
}

Category MlpProblem::category()
{
    return this->data->get_category();
}