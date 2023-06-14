#include "dataset.hpp"
#include "adam.hpp"
#include "pso.hpp"
#include "rmsprop.hpp"



class MlpProblem:public Problem
{
    private:
        Dataset *data;
        int nodes;
        map <int,Data> weights;
        string weight_init;
        mt19937 eng;

    public:
        string saved_path_component;

        MlpProblem(Dataset *d,int n,string weight_initialization_technique);
        ~MlpProblem();

        void set_weights(map <int,Data> &w);
        void set_weights(Data &x);
        void set_nodes(int units);
       

        map <int,Data> get_weights()const;
        string get_weight_init()const;
        int get_nodes()const;
        Data get_sample();

        double minimize_function(Data &x);
        Data gradient(Data &x);
        double sigmoid(double x);
        double sigmoid_derivative(double &x);
        double output(Data &x);
        Data get_derivative(Data &x);
        void optimize_weights(string optimizer);


        double categorical_crossentropy();
        double rmse();
        double mse();
        double get_train_error();
        double get_test_error(Dataset *test_dt);
        Category category();

        vector <pair <double,double>> predict(Dataset *test_dt);
};