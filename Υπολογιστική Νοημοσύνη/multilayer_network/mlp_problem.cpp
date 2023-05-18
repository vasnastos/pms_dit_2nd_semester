#include "mlp_problem.hpp"

MLP::MLP(vector <int> &ls,string wit_method):layers(ls),weight_init_method(wit_method) {
    int num_layers=this->layers.size();
    int previous_layer_size;
    int current_layer_size;

    // Initialize weights and gradients for each layers
    for(int i=1;i<num_layers;i++)
    {
        previous_layer_size=this->layers[i-1];
        current_layer_size=this->layers[i];

        this->weights[i]=this->get_sample(previous_layer_size*current_layer_size);
        this->gradients[i].resize(previous_layer_size*current_layer_size);
        fill(this->gradients[i].begin(),this->gradients[i].end(),0.0);
    }
}

MLP::~MLP() {}

Data MLP::get_sample(int dimension)
{
    // get weight boundaries
}

int MLP::layers_size()const {return this->layers.size();}

double MLP::sigmoid(double &x) {
    return 1.0/(1.0+exp(-x));
}

double MLP::sigmoid_derivative(double &x) {
    auto sig_value=this->sigmoid(x);
    return sig_value * (1-sig_value);
}

void MLP::set_weights(Data &x)
{

}


Data MLP::forward_pass(const Data &input)
{

}

void MLP::backward_pass(const Data &input,const Data &targets)
{

}

void MLP::train(Dataset *train_dt)
{

}

Data MLP::predict(Dataset *test_dt)
{

}