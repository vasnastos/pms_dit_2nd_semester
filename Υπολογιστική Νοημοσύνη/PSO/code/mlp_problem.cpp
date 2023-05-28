#include "mlp_problem.hpp"

MlpProblem::MlpProblem(Dataset *d,int n,string weight_initialization_technique):data(d),nodes(n),Problem((d->dimension()+2)*n) {
    for(int i=0;i<n;i++)
    {
        this->weights[i].resize(d->dimension()+2);
    }
    this->left_margin.resize(this->dimension);
    this->right_margin.resize(this->dimension);
    double lower_bound,upper_bound;


    if(weight_initialization_technique=="Default")
    {
        lower_bound=-10;
        upper_bound=10;
    }
    else if(weight_initialization_technique=="Random")
    {
        lower_bound=-0.01;
        upper_bound=0.01;
    }
    else if(weight_initialization_technique=="Xavier")
    {
        lower_bound=-1/sqrt(d->dimension());
        upper_bound=1/sqrt(d->dimension());
    }
    else if(weight_initialization_technique=="UXavier")
    {
        lower_bound=-6/sqrt(d->dimension()+n);
        upper_bound=6/sqrt(d->dimension()+n);
    }
    else if(weight_initialization_technique=="PSO")
    {
        Data left_mergin;
        Data right_margin;
        left_margin.resize(this->dimension);
        right_margin.resize(this->dimension);
        fill(this->left_margin.begin(),this->left_margin.end(),-10.0);
        fill(this->right_margin.begin(),this->right_margin.end(),10.0);


        this->set_left_margin(left_margin);
        this->set_right_margin(right_margin);

        // PSO optimization in weights
        PSO solver(this,5000,500);
        solver.solve();
        Data best_weights=solver.get_best_x();
        this->set_weights(best_weights);
        return;
    }
    else
    {
        uniform_real_distribution <double> random_lower_bound(-5,-20);
        uniform_real_distribution <double> random_upper_bound(5,20);
        
        lower_bound=random_lower_bound(this->eng);
        upper_bound=random_upper_bound(this->eng);
    }


    // set left and right margin to [-10,10]
    for(int i=0;i<this->dimension;i++)
    {
        this->left_margin[i]=lower_bound;
        this->right_margin[i]=upper_bound;
    }
}

MlpProblem::MlpProblem():nodes(0) {}

MlpProblem::~MlpProblem() {}

// Setters
void MlpProblem::load(string filepath,int nodes,string wit)
{
    this->data=new Dataset;
    this->data->read(filepath,",");
    this->set_nodes(nodes);
    this->set_dimension((this->data->dimension()+2)*nodes);
    double lower_bound,upper_bound;

    if(wit=="Default")
    {
        lower_bound=-10;
        upper_bound=10;
    }
    else if(wit=="Random")
    {
        lower_bound=-0.01;
        upper_bound=0.01;
    }
    else if(wit=="Xavier")
    {
        lower_bound=-1/sqrt(this->data->dimension());
        upper_bound=1/sqrt(this->data->dimension());
    }
    else if(wit=="UXavier")
    {
        lower_bound=-6/sqrt(this->data->dimension()+this->nodes);
        upper_bound=6/sqrt(this->data->dimension()+this->nodes);
    }
    else
    {
        uniform_real_distribution <double> lower_bound_eng(-10,-5);
        uniform_real_distribution <double> upper_bound_eng(5,10);
        lower_bound=lower_bound_eng(this->eng);
        upper_bound=upper_bound_eng(this->eng);
    }


    // set left and right margin to [-10,10]
    for(int i=0;i<this->dimension;i++)
    {
        this->left_margin[i]=lower_bound;
        this->right_margin[i]=upper_bound;
    }
}

void MlpProblem::set_weights(map <int,Data> &w) {this->weights=w;}

void MlpProblem::set_weights(Data &w)
{
    assert(w.size()==this->dimension);
    for(int i=0;i<this->nodes;i++)
    {
        this->weights[i].resize(this->data->dimension()+2);
    }
    
    int weight_size=this->data->dimension()+2;
    int node_i,pos;

    for(int i=0;i<this->dimension;i++)
    {
        node_i=i/weight_size;
        pos=i%weight_size;
        this->weights[node_i][pos]=w[i];
    }
}


void MlpProblem::set_nodes(int units)
{
    this->nodes=units;
}


void MlpProblem::flush()
{
    delete this->data;
}


// Getters
map <int,Data> MlpProblem::get_weights()const {return this->weights;}
 
int MlpProblem::get_nodes()const
{
    return this->nodes;
}


double MlpProblem::minimize_function(Data &w)
{
    this->set_weights(w);
    return this->get_train_error();
    
    // if(this->data->get_category()==Category::CLF)
    // {
    //     return this->categorical_crossentropy();
    // }
    // else if(this->data->get_category()==Category::REG)
    // {
    //     return this->mse();
    // }
}

Data MlpProblem::gradient(Data &x)
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

    // Removed it for categorical crossentropy
    for(int j=0,js=g.size();j<js;j++)
    {
        g[j]*=2.0;
    }
    return g;
}

double MlpProblem::sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
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

    for(int i=0;i<this->nodes;i++)
    {
        dot_product=0.0;
        for(int j=1;j<=d;j++)
        {
            dot_product+=this->weights[i][j]*x[j-1];
        }
        dot_product+=this->weights[i][d+1];
        model_output+=this->weights[i][0] * this->sigmoid(dot_product);
    }
    return model_output;
}

Data MlpProblem::get_derivative(Data &x)
{   
    double dot_product;
    int d=this->data->dimension();
    Data G;
    G.resize(this->dimension);

    for(int i=1;i<=this->nodes;i++)
    {
        dot_product=0.0;
        for(int j=1;j<=d;j++)
        {
            dot_product+=this->weights[i-1][j]*x[j-1];
        }
        dot_product+=this->weights[i-1][d+1];
        G[(d+2)*i-1]=this->weights[i-1][0]*this->sigmoid_derivative(dot_product);//activation
        G[(d+2)*i-(d+1)-1]=this->sigmoid(dot_product);//bias

        for(int k=1;k<=d;k++)
        {
            G[(d+2)*i-(d+1)+k-1]=x[k-1] * this->sigmoid_derivative(dot_product) * this->weights[i-1][0];
        }
    }
    return G;
}

double MlpProblem::get_train_error()
{
    double error=0.0;
    Data xi_point;
    double predicted_value;
    if(this->data->get_category()==Category::CLF)
    {
        for(int i=0,t=this->data->count();i<t;i++)
        {
            xi_point=this->data->get_xpointi(i);
            predicted_value=this->output(xi_point);
            error+=(fabs(this->data->get_class(predicted_value)-this->data->get_class(i))>1e-4);
        }
        error=(error*100.0)/this->data->count();
    }
    else{
        for(int i=0,t=this->data->count();i<t;i++)
        {
            xi_point=this->data->get_xpointi(i);
            predicted_value=this->output(xi_point);
            error+=(predicted_value-this->data->get_ypointi(i));
        }
    }
    return error;
}


double MlpProblem::get_test_error(Dataset *test_dt)
{
    double error=0.0,predicted_value;
    Data xi_point;
    if(this->data->get_category()==Category::CLF)
    {
        double actual_class,predicted_class;
        for(int i=0,t=test_dt->count();i<t;i++)
        {
            xi_point=test_dt->get_xpointi(i);
            actual_class=test_dt->get_class(i);
            predicted_value=this->output(xi_point);
            predicted_class=test_dt->get_class(predicted_value);
            error+=(fabs(predicted_class-actual_class)>1e-4);
        }
        error=(error*100.0)/test_dt->count();
    }
    else
    {
        double actual_value;
        for(int i=0,t=test_dt->count();i<t;i++)
        {
            xi_point=test_dt->get_xpointi(i);
            predicted_value=this->output(xi_point);
            actual_value=test_dt->get_ypointi(i);
            error+=(predicted_value-actual_value);
        }
    }
    return error;
}

void MlpProblem::pso_training()
{
    PSO pso_optimizer(this,5000,500);
    pso_optimizer.solve();

    Data pso_best_weights=pso_optimizer.get_best_x();
    cout<<"==== PSO Optimization in MLP ===="<<endl;
    cout<<"Particles:"<<pso_optimizer.get_particle_count()<<"\tIterations:"<<pso_optimizer.get_max_iters()<<endl;
    cout<<"Best Weights found(Score:"<<pso_optimizer.get_best_y()<<")=>[";
    for(auto &x:pso_best_weights)
    {
        cout<<x<<" ";
    }
    cout<<"]"<<endl<<endl;

    this->set_weights(pso_best_weights);
}


double MlpProblem::categorical_crossentropy()
{
    double loss=0;
    double actual_class,predicted_value,predicted_class;
    Data xi_point;
    for(int i=0;i<this->data->count();i++)
    {
        xi_point=this->data->get_xpointi(i);
        actual_class=this->data->get_class(i);
        predicted_value=this->output(xi_point);
        loss-=actual_class * log(predicted_value)+(1-actual_class) * log(1-predicted_value);
    }
    return loss;
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
    return (error*100.0)/static_cast<double>(this->data->count());
}

string MlpProblem::description()
{
    return this->data->id;
}
