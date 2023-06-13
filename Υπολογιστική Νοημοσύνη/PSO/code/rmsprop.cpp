#include "rmsprop.hpp"

RMSPROP::RMSPROP(Problem *instance):problem(instance),rho(0.9),epsilon(1e-8),iter(0),max_iters(100000) {
    this->squared_gradients.resize(instance->get_dimension());
    this->learning_rate.resize(instance->get_dimension());
    fill(this->squared_gradients.begin(),this->squared_gradients.end(),0.0);
    fill(this->learning_rate.begin(),this->learning_rate.end(),1e-2);
    
    // Possible work with the number of the iterations
    this->xpoint=this->problem->get_sample();
}

RMSPROP::~RMSPROP() {}

void RMSPROP::solve() {
    Data gradient_points;
    do
    {
        gradient_points=this->problem->gradient(this->xpoint);
        for(size_t i=0,size=this->problem->get_dimension();i<size;i++)
        {
            this->squared_gradients[i]=rho * this->squared_gradients[i] + (1.0 - this->rho) * pow(gradient_points[i],2);
            this->learning_rate[i]=this->learning_rate[i]/(this->epsilon+std::sqrt(this->squared_gradients[i]));
            this->xpoint[i]=this->xpoint[i] - this->learning_rate[i] *  gradient_points[i]; 
        }
        this->iter++;
        this->objective_value=this->problem->minimize_function(this->xpoint);
        cout<<"ITER ID:"<<this->iter<<"\tObjective(Error):"<<this->objective_value;
    }while(!this->terminated());
}

void RMSPROP::save(string filename)
{
    fs::path filepath;
    for(const string &x:{"..","results","train_error"})
    {
        filepath.append(x);
    }
    filepath.append(filename);

    fstream fp;
    fp.open(filepath.string(),ios::out);
    for(auto &y_value:this->y_distribution)
    {
        fp<<y_value<<endl;
    }
    fp.close();
}   

Data RMSPROP::get_best_x()const
{
    return this->xpoint;
}

bool RMSPROP::terminated()
{
    double gradient_mean_square=this->problem->grms(this->xpoint);
    cout<<"\tGRMS:"<<gradient_mean_square<<endl;
    return gradient_mean_square<1e-3 || this->iter>=this->max_iters;
}