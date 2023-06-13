#include "mlp_problem.hpp"

bool containsNaN(const std::vector<double>& vec) {
    for (const double& value : vec) {
        if (std::isnan(value)) {
            return true;
        }
    }
    return false;
}

int main(int argc,char **argv)
{
    Config::datasets_db_config();
    string filename="ionosphere.data";

    fs::path path_to_file;
    for(auto &x:{"..","datasets"})
    {
        path_to_file.append(x);
    }
    path_to_file.append(filename);


    Dataset *dataset=new Dataset;
    dataset->read(path_to_file.string());
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


    Dataset train_dt,test_dt;
    train_dt=split_data.first;
    test_dt=split_data.second; 

    train_dt.normalization("standardization");
    test_dt.normalization("standardization");

    MlpProblem model(&train_dt,10,"Xavier");
    model.optimize_weights("PSO");

    cout<<"Test Error:"<<model.get_test_error(&test_dt);


    delete dataset;
    return EXIT_SUCCESS;

}