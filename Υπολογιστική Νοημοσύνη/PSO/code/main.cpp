#include "mlp_problem.hpp"

struct Solution
{
    string id;
    string weight_init;
    string normalization;
    string optimizer;
    double test_error;
    double accuracy;
    Solution(string sid,string wit,string norm,string opt_val,double test_error,double acc):id(sid),weight_init(wit),normalization(norm),optimizer(opt_val),test_error(test_error),accuracy(acc) {}
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
            fs::path pth(".");
            for(const string &x:{"..","results","arena.csv"})
            {
                pth.append(x);
            }

            fstream fp;
            this->arena_file=pth.string();
            fp.open(this->arena_file,ios::out);
            fp<<"Dataset,Weight Init,Normalization,Optimizer,Test Error,Accuracy"<<endl;
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
            vector <string> normalization_methods={"min_max","standardization"};
            vector <string> weight_init_methods={"Random","Xavier","UXavier"};
            vector <string> optimizers={"Adam","PSO"};
            
            Dataset *dataset,train_dt,test_dt;
            double error;
            int experiment_id=1;

            for(const string &x:normalization_methods)
            {
                dataset=new Dataset;
                dataset->read(filename);
                dataset->clean_noise();
                pair <Dataset,Dataset> split_data;
                if(dataset->get_category()==Category::CLF)
                {
                    split_data=dataset->stratify_train_test_split(0.5);
                }
                else if(dataset->get_category()==Category::REG)
                {
                    split_data=dataset->train_test_split(0.5);
                }

                train_dt=split_data.first;
                test_dt=split_data.second;

                train_dt.normalization(x);
                test_dt.normalization(x);

                // for(const string &wit:weight_init_methods)
                // {
                MlpProblem *model=new MlpProblem(&train_dt,10,"");
                for(const string &optimizer:optimizers)
                {
                    cout<<"Id:"<<experiment_id<<" Dataset:"<<dataset->get_id()<<"  Dimension:"<<model->get_dimension()<<"  Normalization:"<<x<<"  TrainMethod:"<<optimizer<<endl;
                    experiment_id++;
                    stringstream filepath;
                    filepath<<train_dt.get_id()<<"_"<<x<<"_"<<"_"<<optimizer<<".wdtrain";
                    model->saved_path_component=filepath.str();
                    model->optimize_weights(optimizer);
                    error=model->get_test_error(&test_dt);
                    this->save(Solution(train_dt.get_id(),"Default",x,optimizer,error,100.0-error));
                }
                // }
                delete model;
                delete dataset;
            }
        }

        void save(const Solution &sol)
        {
            fstream fp;
            fp.open(this->arena_file,ios::app);
            if(!fp.is_open())
            {
                cerr<<"Error in file:"<<this->arena_file<<endl;
                return;
            }
            fp<<sol.id<<","<<sol.weight_init<<","<<sol.normalization<<","<<sol.optimizer<<","<<sol.test_error<<","<<sol.accuracy<<endl;
            fp.close();
        }

        void plot()
        {
            fs::path plot_path;
            for(const string &x:{"..","results","plots"})
            {
                plot_path.append(x);
            }

            string command="python plots.py "+plot_path.string();
            std::system(command.c_str());
            cout<<"Results plotted at "<<plot_path.string()<<endl;
        }
};

int main(int argc,char *argv[])
{
    Config::datasets_db_config();
    Arena arena;
    cout.precision(8);

    for(const string &dataset_name:Config::datasets)
    {
        arena.entrance(dataset_name);
    }

    return 0;
}   