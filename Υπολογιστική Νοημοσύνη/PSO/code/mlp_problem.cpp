#include "mlp_problem.hpp"

MlpProblem::MlpProblem(Dataset *d,int n):data(d),nodes(n),eng(mt19937(high_resolution_clock::now())),Problem((d->dimension()+2)*n) {
    for(int i=0;i<this->nodes;i++)
    {
        this->weights[i].resize(d->dimension()+2);
    }
    this->left_margin.resize(this->dimension);
    this->right_margin.resize(this->dimension);
}
MlpProblem::~MlpProblem() {}

void MlpProblem::set_weights(map <int,Data> &w) {this->weights=w;}

void MlpProblem::set_weight_init(string weight_init_value)
{
    this->weight_init=weight_init_value;
}

void MlpProblem::set_nodes(int units)
{
    this->nodes=units;
}

map <int,Data> MlpProblem::get_weights()const {return this->weights;}

string MlpProblem::get_weight_init()const
{
    return this->weight_init;
}
 
int MlpProblem::get_nodes()const
{
    return this->nodes;
}

Data MlpProblem::get_sample()
{
    uniform_real_distribution <double> rand_real;
    if(this->weight_init=="Random")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-0.01,0.01));
    }
    else if(this->weight_init=="Xavier")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-1/sqrt(this->data->dimension()),1/sqrt(this->data->dimension())));
    }
    else if(this->weight_init=="UXavier")
    {
        rand_real.param(uniform_real_distribution<double>::param_type(-6/sqrt(this->data->dimension()+this->nodes),6/sqrt(this->data->dimension()+this->nodes)));
    }

    for(int i=0;i<this->nodes;i++)
    {
        this->weights[i].resize(this->data->dimension()+2);
        for(int j=0,t=this->weights[i].size();j<t;j++)
        {
            this->weights[i][j]=rand_real(this->eng);
        }
    }
}

double MlpProblem::minimize_function(map <int,Data> &w)
{

}

Data MlpProblem::gradient(map <int,Data> &x)
{
    Data g;
    this->set_weights(x);
    g.resize(this->dimension);
    fill(g.begin(),g.end(),0.0);
    double model_output;

    for(int i=0,t=this->data->count();i<t;i++)
    {
        Data xi_point=this->data->get_xpointi(i);
        Data gradient_points=this->get_derivative(xi_point);
        model_output=this->output(xi_point)-this->data->get_ypointi(i);
        for(int j=0,js=g.size();j<js;j++)
        {
            g[j]+=gradient_points[j]*model_output;
        }
    }

    for(int j=0,js=x.size();j<js;j++)
    {
        g[j]*=2.0;
    }
    return g;
}

double MlpProblem::sigmoid(double x)
{
    return 1.0/(1.0+exp(-1.0));
}

double MlpProblem::sigmoid_derivative(double &x)
{
    double sigmoid_res=this->sigmoid(x);
    return sigmoid_res*(1-sigmoid_res);
}

double MlpProblem::output(Data &x)
{
    double dot_product=0.0;
    double model_output=0.0;
    int d=this->data->dimension();

    for(int i=1;i<=this->nodes;i++)
    {
        dot_product=0.0;
        for(int j=1;j<=d;j++)
        {
            dot_product+=this->weights[i-1][j]*x[j-1];
        }
        dot_product+=this->weights[i-1][d+1];
        model_output+=this->weights[i-1][d+1] * this->sigmoid(dot_product);
    }
    return model_output;
}

Data MlpProblem::get_derivative(Data &x)
{   
    double dot_product;
    int d=this->data->dimension();
    Data G;
    G.resize((d+2)*this->nodes);

    for(int i=1;i<=this->nodes;i++)
    {
        dot_product=0.0;
        for(int j=1;j<=d;j++)
        {
            dot_product+=this->weights[i-1][j]*x[j-1];
        }
        dot_product+=this->weights[i-1][d+1];
        G[(d+2)*i-1]=this->weights[i][0]*this->sigmoid_derivative(dot_product);
        G[(d+2)*i-(d+1)-1]=this->sigmoid(dot_product);

        for(int k=1;k<=d;k++)
        {
            G[(d+2)*i-(d+1)+k-1]=x[k-1]*this->sigmoid_derivative(dot_product) * this->weights[i-1][0];
        }
    }
    return G;
}

double MlpProblem::binary_crossentropy()
{
    double total_loss=0;
    double predicted_value,predicted_class,actual_class;
    Data xi_point;
    for(int i=0,t=this->data->count();i<t;i++)
    {
        xi_point=this->data->get_xpointi(i);
        actual_class=this->data->get_class(i);
        predicted_value=this->output(xi_point);
        predicted_class=this->data->get_class(predicted_value);
        total_loss+=-(actual_class * log(predicted_class)+(1-actual_class)*log(1-predicted_class));
    }
    return total_loss;
}

double MlpProblem::sparse_categorical_crossentropy()
{
    double total_loss=0.0;
    Data xi_point;
    double predicted_value,predicted_class,actual_class;
    for(int i=0,t=this->data->count();i<t;i++)
    {
        xi_point=this->data->get_xpointi(i);
        predicted_value=this->output(xi_point);
        actual_class=this->data->get_class(i);
        total_loss+=actual_class*log(predicted_value);
    }
    return total_loss;
}

double MlpProblem::rmse()
{
    double error=0.0,predicted_value,actual_value;
    Data xi_point;
    for(int i=0,t=this->data->count();i<t;i++)
    {
        xi_point=this->data->get_xpointi(i);
        actual_value=this->data->get_ypointi(i);
        predicted_value=this->output(xi_point);
        error+=pow(actual_value-predicted_value,2);
    }
    return sqrt(error/this->data->count());
}

double MlpProblem::mse()
{
    double error=0.0,predicted_value,actual_value;
    Data xi_point;
    for(int i=0,t=this->data->count();i<t;i++)
    {
        xi_point=this->data->get_xpointi(i);
        actual_value=this->data->get_ypointi(i);
        predicted_value=this->output(xi_point);
        error+=pow(actual_value-predicted_value,2);
    }
    return error/this->data->count();
}

double MlpProblem::get_train_error()
{

}

