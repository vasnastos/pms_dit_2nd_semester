#include "dataset.hpp"

class MLP
{
    private:
        map <int,Data> weights;
        map <int,Data> gradients;
        map <int,Data> activations;
        
        string weight_init_method;

        vector <int> layers;
    
    public:
        MLP(vector <int> &ls,string wit_method);
        ~MLP();

        int layers_size()const;

        double sigmoid(double &x);
        double sigmoid_derivative(double &x);

        void set_weights(Data &x);

        Data get_sample(int dimension);
        Data forward_pass(const Data &input);
        void backward_pass(const Data &input,const Data &targets);

        void train(Dataset *train_dt);
        Data predict(Dataset *test_dt);
};