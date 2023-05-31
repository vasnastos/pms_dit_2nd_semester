#include "adam.hpp"

Adam::Adam(Problem *instance):problem(instance)
{
    this->alpha=1e-3;
    this->beta1=0.9;
    this->beta2=0.999;
    this->learning_rate=1e-2;

    int pdimension=instance->get_dimension();
    this->m.resize(pdimension);
    this->u.resize(pdimension);
    this->mhat.resize(pdimension);
    this->uhat.resize(pdimension);
}

Adam::~Adam() {}

void Adam::solve()
{
    int pdimension=this->problem->get_dimension();
    int iter_id=0;
    this->xpoint=this->problem->get_sample();
    
    // Initialize first_momentum, second_memontum
    fill(this->m.begin(),this->m.end(),0.0);
    fill(this->u.begin(),this->u.end(),0.0);

    Data gradient_points;
    double gradient_mean_square_error;
    while(true)
    {
        gradient_points=this->problem->gradient(this->xpoint);
        for(int i=0;i<pdimension;i++)
        {
            this->m[i]=this->beta1 * this->m[i]+(1-this->beta1) * gradient_points[i];
            this->u[i]=this->beta2 * this->u[i]+(1-this->beta2) * pow(gradient_points[i],2);
            this->mhat[i]=this->m[i]/(1-pow(this->beta1,iter_id+1));
            this->uhat[i]=this->u[i]/(1-pow(this->beta2,iter_id+1));

            this->xpoint[i]=this->xpoint[i] - this->alpha * this->mhat[i]/(sqrt(this->uhat[i])+this->learning_rate);
            // this->xpoint[i]=this->xpoint[i] - this->alpha * this->m[i]/sqrt(this->v[i]+this->learning_rate)
        }
        // this->alpha=this->alpha * sqrt(1-pow(this->beta2,iter_id+1))/(1-pow(this->beta1,iter_id+1));
        iter_id++;
        this->objective_value=this->problem->minimize_function(this->xpoint);
        this->y_distribution.emplace_back(this->objective_value);
        cout.precision(4);
        gradient_mean_square_error=this->problem->grms(this->xpoint);
        cout<<"ADAM. Iter:"<<iter_id<<"\tValue:"<<this->objective_value<<"\tGrms:"<<gradient_mean_square_error<<endl;
        if(gradient_mean_square_error<1e-3)
        {
            break;
        }
    }
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

void Adam::save(string filename)
{
    fstream fp;

    fs::path result_path;
    for(const string &x:{"..","results","train_error"})
    {
        result_path.append(x);
    }
    result_path.append(filename);

    fp.open(result_path.string(),ios::out);
    for(const auto &y_value:this->y_distribution)
    {
        fp<<y_value<<endl;
    }
    fp.close();
}