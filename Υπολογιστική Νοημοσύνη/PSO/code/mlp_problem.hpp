#include "problem.h"
#include "dataset.h"

enum class Category
{
    CLF,
    REG
};

class MlpProblem:public Problem
{
    private:
        Dataset *data;
        int nodes;
        map <int,Data> weights;
        string weight_init;
        Category category;
        mt19937 eng;
    public:
        MlpProblem(Dataset *d,int n);
        ~MlpProblem();

        void set_weights(map <int,Data> &w);
        void set_weight_init(string weight_init_value);
        void set_nodes(int units);
        void set_category(Category &cat);

        map <int,Data> get_weights()const;
        string get_weight_init()const;
        Category get_category()const;
        string get_named_category()const;
        int get_nodes()const;
        Data get_sample();

        double minimize_function(map <int,Data> &x);
        Data gradient(map <int,Data> &x);
        double sigmoid(double x);
        double sigmoid_derivative(double &x);
        double output(Data &x);
        Data get_derivative(Data &x);

        double sparse_categorical_crossentropy();
        double rmse();
        double mse();
        double get_train_error();
};