#include "mlp_problem.hpp"


struct Solution
{
    string id;
    string weight_init;
    string normalization;
    double accuracy;
    Solution(string sid,string wit,string norm,double acc):id(sid),weight_init(wit),normalization(norm),accuracy(acc) {}
};

typedef vector <Solution> SolutionPool;
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

        void solve(string filename,string normalization)
        {
            SolutionPool sols;
            Dataset *dataset=new Dataset;
            dataset->read(Config::get_path(filename),",");
            dataset->normalization(normalization);
            pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.3);
            Dataset train_dt=split_data.first;
            Dataset test_dt=split_data.second;
            
            for(const string &wit:{"Default","Random","Xavier","UXavier","PSO"})
            {
                MlpProblem solver(&train_dt,10,wit);
                double test_error=solver.get_test_error(&test_dt);
                cout<<train_dt.get_id()<<" "<<normalization<<" "<<wit<<"::"<<test_error<<endl;
                sols.emplace_back(Solution(train_dt.get_id(),wit,normalization,test_error));
            }
            this->save_results(sols);
        }

        void save_results(vector <Solution> &solution_pool)
        {
            fstream fp;
            fp.open(this->results_path,ios::app);
            if(!fp.is_open())
            {
                cerr<<"File:"<<this->results_path<<endl;
                return;
            }
            for(auto &sol:solution_pool)
            {
                fp<<sol.id<<sol.weight_init<<sol.normalization<<sol.accuracy<<endl;
            }
            fp.close();
        }
};


int main(int argc,char *argv[])
{
    PlayGround pg;
    Config::datasets_db_config();
    for(const string &dataset_name:Config::datasets)
    {
        for(const string &norm:{"min_max","standardization"})
        {
            pg.solve(dataset_name,norm);
        }
    }
    return 0;
}   