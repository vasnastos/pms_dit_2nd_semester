#include "mlp_problem.hpp"

struct Solution
{
    string id;
    string weight_init;
    string normalization;
    double accuracy;
    Solution(string sid,string wit,string norm,double acc):id(sid),weight_init(wit),normalization(norm),accuracy(acc) {}
};



class PlayGround
{
    
    private:
        string results_path;
    
    public:
        static map <string,Category> datasetsdb;
        PlayGround()
        {

            // Create results file
            fs::path pth(".");
            for(const string &x:{"..","results","arena.csv"})
            {
                pth.append(x);
            }

            fstream fp;
            this->results_path=pth.string();
            fp.open(this->results_path,ios::out);
            fp<<"Dataset,Weight Init,Normalization,Accuracy"<<endl;
            fp.close();
        }

        ~PlayGround() {}

        void solve(string filename)
        {
            Dataset *dataset=new Dataset;
            dataset->read(Config::get_path(filename),",");
            pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.3);
            Dataset train_dt=split_data.first;
            Dataset test_dt=split_data.second;
            MlpProblem solver(&train_dt,10,"PSO");
            double test_error=solver.get_test_error(&test_dt);
            cout<<train_dt.get_id()<<"\t"<<test_error<<endl;
        }


        void save_results(vector <Solution> &solution_pool)
        {
            fstream fp;
            fp.open(this->results_path,ios::app);
            for(auto &sol:solution_pool)
            {
                fp<<sol.id<<sol.weight_init<<sol.normalization<<sol.accuracy<<endl;
            }
            fp.close();
        }
};


int main(int argc,char *argv[])
{
    Config::datasets_db_config();

    int nodes=10;
    string dataset_name="phising_websites.arff";
    string wit="Default";
    string norm="min_max";

    Dataset *dataset=new Dataset;
    dataset->read(Config::get_path(dataset_name),",");
    cout<<*dataset<<endl;
    dataset->normalization(norm);

    // train-test split
    pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.3);
    Dataset train_dt=split_data.first;
    Dataset test_dt=split_data.second;

    MlpProblem solver(&train_dt,nodes,wit);
    auto weights=solver.get_sample();
    solver.set_weights(weights);


    cout<<"Train Error:"<<solver.get_train_error()<<endl;
    cout<<"Test Error:"<<solver.get_test_error(&test_dt)<<endl;
    delete dataset;
    return 0;
}   