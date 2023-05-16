#include "problem.hpp"
#include "dataset.hpp"



class MlpProblem:public Problem
{
    private:
        Dataset *data;
        int nodes;
        map <int,Data> weights;// Weight set per node

    public:
        MlpProblem(Dataset *d,int n,string weight_initialization_technique="");
        ~MlpProblem();

        void set_weights(map <int,Data> &w);
        void set_weights(Data &w);
        void set_nodes(int units);
       

        map <int,Data> get_weights()const;
        int get_nodes()const;
        Data get_sample();

        double minimize_function(Data &x);
        Data gradient(map <int,Data> &x);
        double sigmoid(double x);
        double sigmoid_derivative(double &x);
        double output(Data &x);
        Data get_derivative(Data &x);

        double categorical_crossentropy();
        double rmse();
        double mse();

        double get_train_error();
        double get_test_error(Dataset *test_dt);
};