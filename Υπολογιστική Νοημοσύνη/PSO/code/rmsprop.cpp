#include "rmsprop.hpp"

RMSPROP::RMSPROP(Problem *instance):problem(instance),rho(0.9),epsilon(1e-8) {
    this->squared_gradients.resize(instance->get_dimension());
    this->learning_rate.resize(instance->get_dimension());
    fill(this->squared_gradients.begin(),this->squared_gradients.end(),0.0);
    fill(this->learning_rate.begin(),this->learning_rate.end(),1e-3);
    // Possible work with the number of the iterations
    this->xpoint=this->problem->get_sample();
}

RMSPROP::~RMSPROP() {}

void RMSPROP::solve() {
    size_t iter_id=1;
    Data gradient_points;
    double gradient_mean_square_error;
    while(true)
    {
        gradient_points=this->problem->gradient(this->xpoint);
        for(size_t i=0,size=this->problem->get_dimension();i<size;i++)
        {
            this->squared_gradients[i]=rho * this->squared_gradients[i] + (1.0 - this->rho) * pow(this->squared_gradients[i],2);
            this->learning_rate[i]=iter_id/(this->epsilon+std::sqrt(this->squared_gradients[i]));
            this->xpoint[i]=this->xpoint[i] - this->learning_rate[i] *  gradient_points[i]; 
        }
        iter_id++;
        this->objective_value=this->problem->minimize_function(this->xpoint);
        gradient_mean_square_error=this->problem->grms(this->xpoint);
        cout<<"ITER ID:"<<iter_id<<"\tObjective(Error):"<<this->objective_value<<"\tGrms:"<<gradient_mean_square_error<<endl;
        if(gradient_mean_square_error<1e-3)
        {
            break;
        }
    }
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