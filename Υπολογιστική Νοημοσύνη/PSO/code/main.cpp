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

            // fstream fp;
            this->arena_file=pth.string();
            // fp.open(this->arena_file,ios::out);
            // fp<<"Dataset,Weight Init,Normalization,Optimizer,Test Error,Accuracy"<<endl;
            // fp.close();

            pth=fs::path();
            for(const string &x:{"..","results"})
            {
                pth.append(x);
            }
            this->results_path=pth.string();
        }

        void entrance(string filename)
        {
            vector <string> normalization_methods={"min_max"};
            vector <string> optimizers={"Adam","PSO"};
            
            Dataset *dataset,train_dt,test_dt;
            double error;
            int experiment_id=1;

            // split the data
            dataset=new Dataset;
            dataset->read(filename);
            dataset->clean_noise();
            pair <Dataset,Dataset> split_data;
            if(dataset->get_category()==Category::CLF)
            {
                split_data=dataset->stratify_train_test_split(0.6);
            }
            else if(dataset->get_category()==Category::REG)
            {
                split_data=dataset->train_test_split(0.6);
            }

            for(const string &x:normalization_methods)
            {

                train_dt=split_data.first;
                test_dt=split_data.second;

                train_dt.normalization(x);
                test_dt.normalization(x);

                MlpProblem *model=new MlpProblem(&train_dt,10);
                for(const string &optimizer:optimizers)
                {
                    cout<<"Id:"<<experiment_id<<" Dataset:"<<dataset->get_id()<<"  Dimension:"<<model->getDimension()<<"  Normalization:"<<x<<"  TrainMethod:"<<optimizer<<endl;
                    experiment_id++;
                    stringstream filepath;
                    filepath<<train_dt.get_id()<<"_"<<x<<"_"<<"_"<<optimizer<<".wdtrain";
                    model->save_distribution_path=filepath.str();
                    model->optimize(optimizer);
                    error=model->getTestError(&test_dt);
                    this->save(Solution(train_dt.get_id(),"SMALLVALUES",x,optimizer,error,100.0-error));
                }

                delete model;
            }
            delete dataset;
        }

        void entrance(Dataset *dt)
        {
            vector <string> normalization_methods={"min_max"};
            vector <string> optimizers={"PSO"};

            pair <Dataset,Dataset> split_data;
            Dataset train_dt,test_dt;
            double error;
            dt->clean_noise();
            if(dt->get_category()==Category::CLF)
            {
                split_data=dt->stratify_train_test_split(0.6);
            }
            else
            {
                split_data=dt->train_test_split(0.6);
            }

            for(const string &x:normalization_methods)
            {
                
                train_dt=split_data.first;
                test_dt=split_data.second;

                train_dt.normalization(x);
                test_dt.normalization(x);
                

                for(auto &optimizer:optimizers)
                {
                    MlpProblem *model=new MlpProblem(&train_dt,10);
                    cout<<" Dataset:"<<dt->get_id()<<"  Dimension:"<<model->getDimension()<<"  Normalization:"<<x<<"  TrainMethod:"<<optimizer<<endl;
                    stringstream filepath;
                    filepath<<train_dt.get_id()<<"_"<<x<<"_"<<"_"<<optimizer<<".wdtrain";
                    model->save_distribution_path=filepath.str();
                    model->optimize(optimizer);
                    error=model->getTestError(&test_dt);
                    this->save(Solution(train_dt.get_id(),"SMALL VALUES",x,optimizer,error,100.0-error)); 
                    cout<<"Category:"<<test_dt.get_named_category()<<" Error:"<<error<<endl;  
                    delete model;
                }
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

        string get_path(string dataset_id)
        {
            fs::path fp;
            for(const string &x:{"..","datasets"})
            {
                fp.append(x);
            }
            fp.append(dataset_id);
            return fp.string();
        }
};

Dataset X2()
{
    // Demonstrate x^2 function
    auto f=[](double &x) {return pow(x,2);};
    mt19937 eng;
    uniform_real_distribution <double> crt(-4,4);
    vector <Data> xpoint;
    Data ypoint;

    for(int i=0;i<200;i++)
    {
        Data xi_point;
        xi_point.emplace_back(crt(eng));
        xpoint.emplace_back(xi_point);
        ypoint.emplace_back(f(xi_point[0]));
    }

    Dataset dt;
    dt.set_id("x^2_dt");
    dt.set_category(Category::REG);
    dt.set_data(xpoint,ypoint);
    return dt;
}

Dataset RosenBrock(int d)
{
    auto f=[](Data &x) {
        double s=0;
        for(int i=0,t=x.size();i<t-1;i++)
        {
            s+=100.0*pow((x[i+1]-pow(x[i],2)),2)+pow(1-x[i],2);
        }
        return s;
    };
    mt19937 eng;
    uniform_real_distribution <double> crt(-10,10);
    vector <Data> xpoint;
    Data ypoint;
    for(int i=0;i<1000;i++)
    {
        Data xi_point;
        for(int j=0;j<d;j++)
        {
            xi_point.emplace_back(crt(eng));
        }
        xpoint.emplace_back(xi_point);
        ypoint.emplace_back(f(xi_point));
    }

    Dataset dt;
    dt.set_id("RosenBrock_dim"+to_string(d)+"_dt");
    dt.set_category(Category::REG);
    dt.set_data(xpoint,ypoint);
    return dt;
}

Dataset Bohachevsky()
{
    auto f=[](Data &x) {
        return pow(x[0],2)+2.0 * pow(x[1],2)-0.3*cos(3*pi*x[0])-0.4*cos(4*pi*x[1])+0.7;
    };
    mt19937 eng;
    uniform_real_distribution <double> crt(-100,100);
    vector <Data> xpoint;
    Data ypoint;
    for(int i=0;i<1000;i++)
    {
        Data xi_point;
        for(int j=0;j<2;j++)
        {
            xi_point.emplace_back(crt(eng));
        }
        xpoint.emplace_back(xi_point);
        ypoint.emplace_back(f(xi_point));
    }
    Dataset dt;
    dt.set_id("Bohachvsky");
    dt.set_category(Category::REG);
    dt.set_data(xpoint,ypoint);
    return dt;
}   

Dataset hartman_3()
{
    vector <vector <double>> alpha={
        {3,10,30},
        {0.1,10,35},
        {3,10,30},
        {0.1,10,35}
    };

    vector <double> c={1,1.2,3,3.2};

    vector <vector <double>> p{
        {0.3689,0.117,0.2673},
        {0.4699,0.4387,0.747},
        {0.1091,0.8732,0.5547},
        {0.03815,0.5743,0.8828}
    };

    auto f=[&](Data &x) {
        double s=0;
        for(int i=0,t=4;i<t;i++)
        {
            double inline_sum=0;
            for(int j=0;j<3;j++)
            {
                inline_sum+=alpha[i][j]*(x[j]-p[i][j]);
            }
            s+=c[i]*exp(-inline_sum);
        }
        return s;
    };
    mt19937 eng;
    uniform_real_distribution <double> crt(0,1);
    vector <Data> xpoint;
    Data ypoint;
    double obj;
    for(int i=0;i<1000;i++)
    {
        Data xi_point;
        for(int j=0;j<3;j++)
        {
            xi_point.emplace_back(crt(eng));
        }
        xpoint.emplace_back(xi_point);
        ypoint.emplace_back(f(xi_point));
    }
    Dataset dt;
    dt.set_id("Hartman_3");
    dt.set_category(Category::REG);
    dt.set_data(xpoint,ypoint);
    return dt;
}


int main(int argc,char *argv[])
{
    Config::datasets_db_config();
    Arena arena;
    cout.precision(12);
    

    Dataset rbrc=RosenBrock(6);
    arena.entrance(&rbrc);

    // Dataset bchy=Bohachevsky();
    // arena.entrance(&bchy);
    
    // Dataset hartman=hartman_3();
    // arena.entrance(&hartman);
    
    // Dataset dt=RosenBrock(7);
    // arena.entrance(&dt);
    
    // for(const string &dataset_name:Config::datasets)
    // {
    //     arena.entrance(dataset_name);
    // }

    // vector <Dataset*> function_learning;
    // Dataset d1=X2();
    // Dataset d2=RosenBrock(6);
    // Dataset d3=RosenBrock(8);
    // function_learning.emplace_back(&d1);
    // function_learning.emplace_back(&d2);
    // function_learning.emplace_back(&d3);

    // for(auto &data:function_learning)
    // {
    //     arena.entrance(data);
    // }

    // for(auto &df:function_learning)
    // {
    //     delete df;
    // }
    // return 0;
}   