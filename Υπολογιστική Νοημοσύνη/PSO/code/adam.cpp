#include "adam.hpp"

Adam::Adam(Problem *instance):problem(instance),iter(0),max_iters(100000)
{
    this->alpha=1e-4;
    this->beta1=0.9;
    this->beta2=0.999;
    this->learning_rate=1e-2;

    int pdimension=instance->getDimension();
    this->m.resize(pdimension);
    this->u.resize(pdimension);
    this->mhat.resize(pdimension);
    this->uhat.resize(pdimension);
}

Adam::~Adam() {}

void Adam::solve()
{
    int pdimension=this->problem->getDimension();
    this->xpoint=this->problem->getSample();
    
    // Initialize first_momentum, second_memontum
    fill(this->m.begin(),this->m.end(),0.0);
    fill(this->u.begin(),this->u.end(),0.0);

    Data gradient_points;
    do
    {
        gradient_points=this->problem->gradient(this->xpoint);

        for(int i=0;i<pdimension;i++)
        {
            this->m[i]=this->beta1 * this->m[i]+(1-this->beta1) * gradient_points[i];
            this->u[i]=this->beta2 * this->u[i]+(1-this->beta2) * pow(gradient_points[i],2);
            this->mhat[i]=1.0/(1-pow(this->beta1,this->iter+1))*this->m[i];
            this->uhat[i]=1.0/(1-pow(this->beta2,this->iter+1))*this->u[i];

            this->xpoint[i]=this->xpoint[i] - this->alpha * this->mhat[i]/(sqrt(this->uhat[i])+this->learning_rate);
            // this->xpoint[i]=this->xpoint[i] - this->alpha * this->m[i]/sqrt(this->v[i]+this->learning_rate)
        }
        // this->alpha=this->alpha * sqrt(1-pow(this->beta2,iter_id+1))/(1-pow(this->beta1,iter_id+1));
        this->objective_value=this->problem->statFunmin(this->xpoint);
        this->iter++;
        this->y_distribution.emplace_back(this->objective_value);
        cout<<"ADAM. Iter:"<<this->iter<<"\tValue:"<<this->objective_value;
    }while(!this->terminated());
}

void Adam::set_alpha(double new_alpha_value) {this->alpha=new_alpha_value;}
double Adam::get_alpha()const {return this->alpha;}
void Adam::set_beta1(double new_beta1_value)  {this->beta1=new_beta1_value;}
double Adam::get_beta1()const {return this->beta1;}
void Adam::set_beta2(double new_beta2_value) {this->beta2=new_beta2_value;}
double Adam::get_beta2()const {return this->beta2;}
void Adam::set_learning_rate(double new_epsilon) {this->learning_rate=new_epsilon;}
double Adam::get_learning_rate()const {return this->learning_rate;}
Data Adam::get_best_x()const {return this->xpoint;}

bool Adam::terminated()
{
    double gradient_mean_square_error=this->problem->grms(this->xpoint);
    cout<<"\tGrms:"<<gradient_mean_square_error<<endl;
    return gradient_mean_square_error<1e-3  || fabs(this->objective_value-0)<1e-4;
}

void Adam::save(string filename)
{
    fstream fp;
    fs::path pth;
    for(auto &x:{"..","results","distribution"})
    {
        pth.append(x);
    }
    pth.append(filename);

    fp.open(pth.string(),ios::out);
    for(const auto &y_value:this->y_distribution)
    {
        fp<<y_value<<endl;
    }
    fp.close();
}