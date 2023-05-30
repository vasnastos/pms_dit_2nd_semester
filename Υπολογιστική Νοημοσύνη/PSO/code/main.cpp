#include "mlp_problem.hpp"


struct Solution
{
    string id;
    string weight_init;
    string normalization;
    double test_error;
    double accuracy;
    Solution(string sid,string wit,string norm,double test_error,double acc):id(sid),weight_init(wit),normalization(norm),test_error(test_error),accuracy(acc) {}
};

class Arena
{
    private:
        string results_path;
        string arena_file;
        vector <Solution> results;
    public:
        Arena() {
            // Create results file
            fs::path pth;
            for(const string &x:{"..","results","arena.csv"})
            {
                pth.append(x);
            }

            fstream fp;
            this->arena_file=pth.string();
            fp.open(this->results_path,ios::out);
            fp<<"Dataset,Weight Init,Normalization,Test Error,Accuracy"<<endl;
            fp.close();

            pth=fs::path();
            for(const string &x:{"..","results"})
            {
                pth.append(x);
            }
            this->results_path=pth.string();
        }

        void entrance(string filename)
        {
            Dataset *dataset,train_dt,test_dt;
            double error;
            int experiment_id=1;

            for(const string &x:{"min-max","standardization"})
            {
                dataset=new Dataset;
                dataset->read(filename);
                dataset->normalization(x);  
                pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.5);
                train_dt=split_data.first;
                test_dt=split_data.second;

                for(const string &wit:{"Random","Xavier","UXavier"})
                {
                    MlpProblem model(&train_dt,10,wit);
                    for(const string &optimizer:{"Adam","PSO"})
                    {
                        cout<<"Id:"<<experiment_id<<"  Dimension:"<<model.get_dimension()<<"  Normalization:"<<x<<"  WeightInit:"<<wit<<"  TrainMethod:"<<optimizer<<endl;
                        model.optimize_weights(optimizer);
                    }
                    error=model.get_test_error(&test_dt);
                    this->results.emplace_back(Solution(train_dt.get_id(),wit,x,error,1.0-error));
                }
                delete dataset;
            }
        }

        void save()
        {
            fstream fp;
            fp.open(this->arena_file,ios::app);
            if(fp.is_open())
            {
                cerr<<"Error in file:"<<this->arena_file<<endl;
                return;
            }
            for(auto &sol:this->results)
            {
                fp<<sol.id<<","<<sol.weight_init<<","<<sol.normalization<<","<<sol.test_error<<","<<sol.accuracy<<endl;
            }
            fp.close();
        }
};


int main(int argc,char *argv[])
{
    Config::datasets_db_config();
    Arena arena;

    for(const string &dataset_name:Config::datasets)
    {
        arena.entrance(dataset_name);
    }

    return 0;
}   