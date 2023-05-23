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
        fs::path filepath;
        string results_path;
    
    public:
        vector <string> datasets;
        PlayGround():filepath(fs::path("."))
        {
            // Get datasets from datasets folder
            for(const string &x:{"..","datasets"})
            {
                this->filepath.append(x);
            }

            for(const auto &entry:fs::directory_iterator(this->filepath))
            {
                this->datasets.emplace_back(entry.path().string());
            }

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

        string get_path(string filename)const
        {
            auto itr=find(this->datasets.begin(),this->datasets.end(),filename);
            if(itr!=this->datasets.end())
            {
                fs::path pth(this->filepath);
                pth.append(filename);
                return pth.string();
            }
            return "";
        }


        void solve(string filename)
        {
            auto filepath=this->get_path(filename);
            if(filepath=="") {return;}
            vector <Solution> dataset_solutions;

            for(const auto &norm:{"min_max","standardization"})
            {
                Dataset *dataset=new Dataset;
                dataset->read(filepath,",");
                dataset->normalization(norm);
                
                pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.3);
                Dataset train_dt=split_data.first;
                Dataset test_dt=split_data.second;

                for(const auto &weight_init:{"Default","Random","Xavier","UXavier"})
                {
                    MlpProblem solver(&train_dt,10,weight_init);
                    solver.pso_training();
                    auto accuracy=solver.get_test_error(&test_dt);
                    dataset_solutions.emplace_back(Solution(dataset->get_id(),weight_init,norm,accuracy));
                    cout<<dataset->get_id()<<"\t"<<weight_init<<"\t"<<norm<<"\t"<<accuracy<<endl;
                }    
                delete dataset;
            }
            this->save_results(dataset_solutions);
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

int main(int argc,char **argv)
{
    PlayGround playground;
    for(const string &dataset:playground.datasets)
    {
        playground.solve(dataset);
    }
}
