#include "../PSO/code/dataset.hpp"

class MLP
{
    private:
        vector <int> layers;
        map <int,Data> weights;
        map <int,Data> gradients;
        map <int,Data> activations;
        string weight_init_method;//UXavier

        string activation;// Potential values(sigmoid,softmax)
        double learning_rate;//1e-3
        int epochs;//1000
        mt19937 eng;

    public:
        MLP(vector <int> &ls,double lr,string wit_method,string activationf,int num_epochs);
        ~MLP();

        int layers_size()const;

        Data softmax(const Data &input);
        double sigmoid(double &x);
        double accuracy_score(Dataset *train_dt,const vector <Data> &predictions);

        Data get_sample(int current_layer_idx);
        Data forward_pass(const Data &input);
        void backward_pass(const Data &input,const Data &targets);
        double dot_product(int node_i,int layer_idx);

        void train(Dataset *train_dt);
        Data predict(Dataset *test_dt);
};