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
        static map <string,Category> datasetsdb;
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

        static void datasets_db_config();

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
                dataset->set_category(PlayGround::datasetsdb[dataset->get_id()]);
                dataset->normalization(norm);
                pair <Dataset,Dataset> split_data=dataset->stratify_train_test_split(0.3);
                Dataset train_dt=split_data.first;
                Dataset test_dt=split_data.second;

                cout<<"Train:"<<train_dt.count()<<"\tTest:"<<test_dt.count()<<endl;
                system("pause");

                for(const auto &weight_init:{"Default","Random","Xavier","UXavier"})
                {
                    cout<<dataset->get_id()<<"\t"<<weight_init<<"\t"<<norm<<endl;
                    MlpProblem solver(&train_dt,10,weight_init);
                    solver.pso_training();
                    auto accuracy=solver.get_test_error(&test_dt);
                    dataset_solutions.emplace_back(Solution(dataset->get_id(),weight_init,norm,accuracy));
                }    
                delete dataset;
            }
            this->save_results(dataset_solutions);
        }

        void solve(int file_index)
        {
            if(this->datasets.empty())
            {
                cerr<<"Datasets container is empty"<<endl;
            }
            if(file_index<=0 || file_index>this->datasets.size())
            {
                cerr<<"File index:"<<file_index<<" does not exist on the dataset-Select one of the following[1-"<<this->datasets.size()<<"]"<<endl;
                return;
            }
            string filename=this->datasets[file_index-1]; 
            this->solve(filename);
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

map <string,Category> PlayGround::datasetsdb=map <string,Category>();

void PlayGround::datasets_db_config()
{
    fs::path pth(".");
    for(const auto &x:{"..","datasets_db.csv"})
    {
        pth.append(x);
    }

    fstream fp;
    fp.open(pth.string(),std::ios::in);

    if(!fp.is_open())
    {
        cerr<<"File did not open properly"<<endl;
        return;
    }

    string line,word;
    vector <string> data;
    bool headers=true;
    while(getline(fp,line))
    {
        if(line=="") continue;

        if(headers)
        {
            headers=false;
            continue;
        }

        data.clear();

        stringstream ss(line);
        while(getline(ss,word,','))
        {
            data.emplace_back(word);
        }

        if(data.size()!=2) continue;

        Category cat;
        if(data[1]=="clf")
        {
            cat=Category::CLF;
        }
        else if(data[1]=="reg")
        {
            cat=Category::REG;
        }

        PlayGround::datasetsdb[data[0]]=cat;
    }
    fp.close();
}


int main(int argc,char **argv)
{
    PlayGround playground;
    PlayGround::datasets_db_config();

    cout<<"----- Datasets -----"<<endl;
    int i=1;
    for(const string &dataset:playground.datasets)
    {
        cout<<i<<">"<<dataset<<endl;
        i++;
    }
    playground.solve(1);
    return 0;
}
