#include "mlp_problem.hpp"

MLP::MLP(vector <int> &ls,double lr,string wit_method,string activationf,int num_epochs):layers(ls),weight_init_method(wit_method),learning_rate(lr),activation(activationf),epochs(num_epochs),eng(mt19937(default_random_engine{})) {
    int num_layers=this->layers.size();
    int previous_layer_size;
    int current_layer_size;

    // Initialize weights and gradients for each layers
    for(int i=1;i<num_layers;i++)
    {
        previous_layer_size=this->layers[i-1];
        current_layer_size=this->layers[i];

        this->weights[i]=this->get_sample(i);
        this->gradients[i].resize(previous_layer_size*current_layer_size);
        fill(this->gradients[i].begin(),this->gradients[i].end(),0.0);
    }

    for(int i=0;i<num_layers;i++)
    {
        this->activations[i].resize(this->layers[i]);
        fill(this->activations[i].begin(),this->activations[i].end(),0.0);
    }
}

MLP::~MLP() {}

Data MLP::get_sample(int current_layer_idx)
{
    double lb;
    double ub;
    if(this->weight_init_method=="Default")
    {
        lb=-10;
        ub=10;
    }
    else if(this->weight_init_method=="Random")
    {
        lb=-0.01;
        ub=0.01;
    }
    else if(this->weight_init_method=="Xavier")
    {
        lb=sqrt(-1.0/this->layers[current_layer_idx]);
        ub=sqrt(1.0/this->layers[current_layer_idx]);
    }
    else
    {
        // uxavier
        lb=sqrt(-6.0/(this->layers[current_layer_idx]*this->layers[current_layer_idx-1]));
        ub=sqrt(6.0/(this->layers[current_layer_idx]*this->layers[current_layer_idx-1]));
    }

    int dimension=this->layers[current_layer_idx]*this->layers[current_layer_idx-1]; 
    uniform_real_distribution <double> rand_real(lb,ub); 

    // Initialize weights
    Data sample_vector;
    sample_vector.resize(dimension);

    for(int i=0,t=sample_vector.size();i<t;i++)
    {
        sample_vector[i]=rand_real(this->eng);
    }
    return sample_vector;
}

int MLP::layers_size()const {return this->layers.size();}

double MLP::sigmoid(double &x) {
    return 1.0/(1.0+exp(-x));
}

Data MLP::softmax(const Data &input)
{
    Data output;

    double max_value=*std::max_element(input.begin(),input.end());
    double sum_exp=0.0;

    for(const auto &value:input)
    {
        double exp_value=std::exp(value-max_value);
        output.emplace_back(exp_value);
        sum_exp+=exp_value;
    }

    for(auto &value:output)
    {
        value/=sum_exp;
    }

    return output;
}


double MLP::dot_product(int node_i,int layer_idx)
{
    double dp=0.0;
    int previous_layer_size=this->weights[layer_idx-1].size()/this->activations[layer_idx-1].size();
    for(int j=0;j<previous_layer_size;j++)
    {
        dp+=this->weights[layer_idx-1][node_i*this->layers[layer_idx]+j] * this->activations[layer_idx-1][j];
    }
    return dp;
}


Data MLP::forward_pass(const Data &input)
{
    this->activations[0]=input;
    int num_layers=this->layers_size();
    int previous_layer_size,current_layer_size;
    double node_dot_product;
    for(int current_layer_idx=1;current_layer_idx<num_layers;current_layer_idx++)
    {
        previous_layer_size=this->layers[current_layer_idx-1];
        current_layer_size=this->layers[current_layer_idx];
        for(int current_node_idx=0;current_node_idx<current_layer_size;current_node_idx++)
        {
            node_dot_product=this->dot_product(current_node_idx,current_layer_idx);
            if(this->activation=="sigmoid")
            {
                this->activations[current_layer_idx][current_node_idx]=this->sigmoid(node_dot_product); 
            }
            else if(this->activation=="softmax")
            {
                this->activations[current_layer_idx][current_node_idx]=node_dot_product;
            }
        }

        if(this->activation=="softmax")
        {
            this->activations[current_layer_idx]=this->softmax(this->activations[current_layer_idx]);
        }

    }
    return this->activations[num_layers-1]; 
}

void MLP::backward_pass(const Data &input,const Data &targets)
{
    int num_layers=this->layers_size();
    int output_idx=num_layers-1;
    int output_layer_size=this->activations[output_idx].size();
    for(int i=0;i<output_layer_size;i++)
    {
        double output=this->activations[output_idx][i];
        double error=output*(1.0-output) * (targets[i]-output); // crossentropy

        for(int j=0,gs=this->activations[output_idx-1].size();j<gs;j++)
        {
            this->gradients[output_idx-1][i*gs+j]=error * this->activations[output_idx-1][j];
        }
    }

    // Calculate gradients for hidden layers
    for(int i=output_idx-1;i>0;i--)
    {
        int current_layer_size=this->activations[i].size();
        int next_layer_size=this->activations[i+1].size();

        for(int j=0;j<current_layer_size;j++)
        {
            double error=0.0;
            double output=this->activations[i][j];
            for(int k=0;k<next_layer_size;k++)
            {
                error+=this->weights[i][k*current_layer_size+j]* this->gradients[i][k * current_layer_size + j];
            }
            this->gradients[i-1][j]=(output) * (1.0-output) * error;  // sigmoid Derivative * dot_product
        }
    }

    // update weights / Batch BackPropagation
    int current_layer_size,next_layer_size;
    for(int i=0;i<num_layers-1;i++)
    {
        current_layer_size=this->activations[i].size();
        next_layer_size=this->activations[i+1].size();
        for(int j=0;j<next_layer_size;j++)
        {
            for(int k=0;k<current_layer_size;k++)
            {
                this->weights[i][j*current_layer_size+k]+=this->learning_rate*this->gradients[i][j*current_layer_size+k] * this->activations[i][k];
            }
        }

    }
}

double MLP::accuracy_score(Dataset *train_dt,const vector <Data> &predictions)
{
    double score=0.0;
    double predicted_class;
    bool is_multioutput=this->layers[this->layers_size()-1]!=1;
    for(int i=0,ps=predictions.size();i<ps;i++)
    {
        if(train_dt->get_category()==Category::CLF && !is_multioutput)
        {
            predicted_class=predictions[i].at(0);
            predicted_class=train_dt->get_class(predicted_class);
        }
        else if(train_dt->get_category()==Category::CLF && is_multioutput)
        {
            predicted_class=static_cast<double>(std::max_element(predictions[i].begin(),predictions[i].end())-predictions[i].begin());
            predicted_class=train_dt->get_class(predicted_class);
        }
        score+=(fabs(predicted_class-train_dt->get_class(i))<=1e-4);
    }

    return (score*100.0)/static_cast<double>(train_dt->count());
}

void MLP::train(Dataset *train_dt)
{
    Data input,targets;
    Data sample_output;
    vector <Data> train_predictions;
    for(int epoch=0;epoch<this->epochs;epoch++)
    {
        for(int sample_idx=0,samples_sum=train_dt->count();sample_idx<samples_sum;sample_idx++)
        {
            input=train_dt->get_xpointi(sample_idx);
            targets.clear();
            targets.emplace_back(train_dt->get_class(sample_idx));
            sample_output=this->forward_pass(input);
            this->backward_pass(input,targets);
            train_predictions.emplace_back(sample_output);
        }
        cout<<"Epoch "<<epoch<<"\t"<<"Accuracy:"<<this->accuracy_score(train_dt,train_predictions)<<endl;
    }
}

Data MLP::predict(Dataset *test_dt)
{

}