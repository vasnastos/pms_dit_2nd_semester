#include "solver.hpp"

Solver::Solver(int number_of_units):units(number_of_units),problem(nullptr)
{
    // 1. load the folder of potential datasets
    this->datasets_path=fs::path(".");
    for(const string &x:{"..","datasets"})
    {
        this->datasets_path.append(x);
    }

    vector <string> split_data;
    string word;

    // 2. get dataset names
    for(const auto &entry : fs::directory_iterator(this->datasets_path))
    {
        if(entry.is_regular_file())
        {
            split_data.clear();
            auto ss= stringstream(entry.path().string());
            while(getline(ss,word,sep))
            {
                split_data.emplace_back(word);
            }
            this->dataset_names.emplace_back(entry.path().string());
        }
    }
}

Solver::~Solver()
{

}

void Solver::load(string filename)
{
    this->flush();
    if(find(this->dataset_names.begin(),this->dataset_names.end(),filename)==this->dataset_names.end())
    {
        cerr<<"File:"<<filename<<" not found on datasets list"<<endl;
        return;
    }

    fs::path full_path=this->datasets_path;
    full_path.append(filename);
    if(this->problem!=nullptr)
    {
        this->problem->flush();
    }
    this->problem=new MlpProblem;
    this->problem->load(full_path.string(),this->units,"");
}

void Solver::flush()
{
    delete this->problem;
    delete this->pso;
}

void Solver::solve()
{

}

Data Solver::get_best_weights()
{

}