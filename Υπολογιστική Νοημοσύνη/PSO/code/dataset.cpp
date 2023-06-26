#include "dataset.hpp"


Dataset::Dataset():id("") {}
Dataset::~Dataset() {}

void Dataset::set_id(const string &dataset_id)
{
    this->id=dataset_id;
}

void Dataset::set_data(vector <Data> &xpoint_set,Data &ypoint_set)
{
    this->xpoint=xpoint_set;
    this->ypoint=ypoint_set;
}

void Dataset::set_category(const Category &cat)
{
    this->category=cat;
}

string Dataset::get_id()const
{
    return this->id;
}

void Dataset::read(string filename)
{
    this->id=Config::get_id(filename);
    string seperator=Config::get_separator(this->id);
    bool has_categorical=Config::categorical_label(this->id);
    this->set_category(Config::get_category(this->id));

    bool header=(this->id=="Concrete_Data" || this->id=="forestfires" || this->id=="RP_hardware_performance");

    fstream fp;
    fp.open(filename,ios::in);
    if(!fp.is_open())
    {
        std::cerr<<"File did not open properly:"<<filename<<endl;
        return;
    }

    vector <string> data;
    string line,word,substring;
    size_t start_pos,seperator_pos;

    if(has_categorical)
    {
        vector <string> labels;
        set <string> distinct_labels;
        int index=0;
        while(getline(fp,line))
        {
            if(header)
            {
                header=false;
                continue;
            }
            if(line=="") continue;
            if(trim(line)[0]=='@') continue;

            data=split(line,seperator);

            // while(seperator_pos!=string::npos)
            // {
            //     if(seperator_pos>line.size())
            //     {
            //         break;
            //     }

            //     substring=line.substr(start_pos,seperator_pos-start_pos);
            //     data.emplace_back(substring);
            //     start_pos=seperator_pos+1;
            //     #ifdef __linux__
            //         if(line[seperator_pos+1]=='\r')
            //         {
            //             start_pos++;
            //         }
            //     #endif
            //     seperator_pos=line.find(start_pos);

            // }
            // data.emplace_back(line.substr(start_pos));

            Data row;
            for(int i=0,t=data.size()-1;i<t;i++)
            {
                row.emplace_back(stod(data.at(i)));
            }
            this->xpoint.emplace_back(row);
            labels.emplace_back(data.at(data.size()-1));
            distinct_labels.insert(data.at(data.size()-1));
        }
        for(const auto &label:labels)
        {
            index=0;
            for(auto &label_val:distinct_labels)
            {
                if(label==label_val) break;
                index++;
            }
            this->ypoint.emplace_back(static_cast<double>(index));
        }
    }
    else
    {
        while(getline(fp,line))
        {
            if(header)
            {
                header=false;
                continue;
            }

            if(line=="") continue;
            if(trim(line)[0]=='@') continue;
            data=split(line,seperator);

            Data row;
            for(int i=0,t=data.size()-1;i<t;i++)
            {
                row.emplace_back(stod(data.at(i)));
            }
            this->xpoint.emplace_back(row);
            this->ypoint.emplace_back(stod(data.at(data.size()-1)));
        }
    }
    fp.close();
    this->make_patterns();
}


int Dataset::dimension()const
{
    if(this->xpoint.empty())
    {
        return 0;
    }
    return this->xpoint.at(0).size();
}

int Dataset::count()const
{
    return this->xpoint.size();
}

double Dataset::xmax(int pos)
{
    if(pos<0 || pos>=this->dimension() || this->xpoint.empty())
    {
        cerr<<"Position Error:"<<pos<<endl;
        return -1.0;
    }

    return std::max_element(this->xpoint.begin(),this->xpoint.end(),[&](const Data &d1,const Data &d2) {return d1.at(pos)<d2.at(pos);})->at(pos);
}

double Dataset::xmin(int pos)
{
    if(pos<0 || pos>=this->dimension() || this->xpoint.size()==0)
    {
        cerr<<"Position Error:"<<pos<<endl;
        return -1.0;
    }
    return std::min_element(this->xpoint.begin(),this->xpoint.end(),[&](const Data &d1,const Data &d2) {return d1.at(pos)<d2.at(pos);})->at(pos);
}

double Dataset::ymax()
{
    if(this->ypoint.empty())
    {
        return -1.0;
    }

    return *std::max_element(this->ypoint.begin(),this->ypoint.end());
}
double Dataset::ymin()
{
    if(this->ypoint.empty())
    {
        return -1.0;
    }
    return *std::min_element(this->ypoint.begin(),this->ypoint.end());
}

double Dataset::stdx(int pos)
{
    if(pos<0 || pos>=this->dimension() || this->xpoint.empty())
    {
        cerr<<"Position Error:"<<pos<<endl;
        return -1.0;
    }

    double pos_mean=this->xmean(pos);
    return sqrt(accumulate(this->xpoint.begin(),this->xpoint.end(),0.0,[&](double &s,const Data &d) {return s+pow(d.at(pos)-pos_mean,2);})/this->count());
}

double Dataset::stdy()
{
    if(this->ypoint.empty())
    {
        return -1.0;
    }
    double y_mean=this->ymean();
    return sqrt(accumulate(this->ypoint.begin(),this->ypoint.end(),0.0,[&](double &s,const double &d) {return s+pow(d-y_mean,2);})/this->count());
}

double Dataset::xmean(int pos)
{
    if(pos<0 || pos>=this->dimension() || this->xpoint.size()==0)
    {
        cerr<<"Position Error:"<<pos<<endl;
        return -1.0;
    }
    return accumulate(this->xpoint.begin(),this->xpoint.end(),0.0,[&](double &s,const Data &d) {return s+d.at(pos);})/this->count();
}   

double Dataset::ymean()
{
    if(this->ypoint.empty())
    {
        return -1.0;
    }
    return accumulate(this->ypoint.begin(),this->ypoint.end(),0.0,[&](double &s,const double &d) {return s+d;})/this->count();
}

void Dataset::normalization(string ntype)
{
    if(ntype=="min_max")
    {
        Data max_data,min_data;
        max_data.resize(this->dimension());
        min_data.resize(this->dimension());
        for(int i=0,t=this->dimension();i<t;i++)
        {
            max_data[i]=this->xmax(i);
            min_data[i]=this->xmin(i);
        }
        double maxy_data=this->ymax(),miny_data=this->ymin();

        for(int i=0,rows=this->count();i<rows;i++)
        {
            for(int j=0,cols=this->dimension();j<cols;j++)
            {
                this->xpoint[i][j]=(this->xpoint[i][j]-min_data[j])/(max_data[j]-min_data[j]);
            }
            this->ypoint[i]=(this->ypoint[i]-miny_data)/(maxy_data-miny_data);
        }
        this->make_patterns();
    }
    else if(ntype=="standardization")
    {
        Data mean_data,std_data;
        mean_data.resize(this->dimension());
        std_data.resize(this->dimension());
        double meany=this->ymean(),ystd=this->stdy();
        for(int i=0,t=this->dimension();i<t;i++)
        {
            mean_data[i]=this->xmean(i);
            std_data[i]=this->stdx(i);
        }

        for(int i=0,rows=this->count();i<rows;i++)
        {
            for(int j=0,cols=this->dimension();j<cols;j++)
            {
                this->xpoint[i][j]=(this->xpoint[i][j]-mean_data[j])/std_data[j];
            }
            this->ypoint[i]=(this->ypoint[i]-meany)/ystd;
        }
        this->make_patterns();
    }
}

Data Dataset::get_xpointi(int pos)
{
    if(pos<0 || pos>=this->count() || this->xpoint.empty())
    { 
        cerr<<"Position error:"<<pos<<endl;
        return Data();
    }
    return this->xpoint.at(pos);
}

double Dataset::get_ypointi(int pos)
{
    if(this->ypoint.empty() || pos<0 || pos>=this->count())
    {
        cerr<<"Position error:"<<pos<<endl;
        return -1.0;
    }
    return this->ypoint.at(pos);
}

void Dataset::make_patterns()
{
    this->patterns.clear();

    for(auto &pattern:this->ypoint)
    {
        if(find_if(this->patterns.begin(),this->patterns.end(),[&](const double &c) {return fabs(pattern-c)<=1e-4;})==this->patterns.end())
        {
            this->patterns.emplace_back(pattern);
        }
    } 
}


Category Dataset::get_category()const {return this->category;}

string Dataset::get_named_category()const
{
    switch (this->category)
    {
        case Category::CLF:
            return "Classification";
            break;
        case Category::REG:
            return "Regression";
            break;
        default:
            return "No-Category";
            break;
    }
}


double Dataset::get_class(double &value)
{
    int imin=-1;
    double dmin=1e+100;
    double diff;
    if(this->category==Category::CLF)
    {
        for(int i=0,t=this->patterns.size();i<t;i++)
        {
            diff=fabs(value-this->patterns[i]);
            if(diff<dmin)
            {
                dmin=diff;
                imin=i;
            }
        }
        
        return this->patterns.at(imin);
    }
    return -20;
}

double Dataset::get_class(int &pos)
{
    if(this->xpoint.empty() || pos<0 || pos>=this->count())
    {
        cerr<<"Position error:"<<pos<<endl;
    }
    double y_value=this->get_ypointi(pos);
    return this->get_class(y_value);
}

int Dataset::no_classes()const
{
    return this->patterns.size();
}

pair <Dataset,Dataset> Dataset::stratify_train_test_split(double test_size)
{
    vector <Data> train_xpoint_set;
    vector <Data> test_xpoint_set;
    Data train_ypoint_set;
    Data test_ypoint_set;

    map <double,int> class_counter;
    for(auto &pattern:this->patterns)
    {
        class_counter[pattern]=std::accumulate(this->ypoint.begin(),this->ypoint.end(),0,[&](int s,const double &y) {return s+(fabs(pattern-y)<=1e-4);});
    }

    map <double,int> train_sizes;
    for(auto &[pattern,count]:class_counter)
    {
        train_sizes[pattern]=count*(1.0-test_size);
    }


    mt19937 eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution <int> rand_int(0,this->count()-1);
    vector <int> test_indeces;
    vector <int> train_indeces;

    for(int sample_idx=0;sample_idx<this->count();sample_idx++)
    {
        test_indeces.emplace_back(sample_idx);
    }

    int index;
    double actual_class;
    bool found;

    while(true)
    {
        found=false;
        do{
            index=rand_int(eng);
            auto itr=find(train_indeces.begin(),train_indeces.end(),index);
            if(itr!=train_indeces.end()) continue;
            actual_class=this->get_class(this->ypoint[index]);
            
            for(auto &[pattern,count]:train_sizes)
            {   
                
                if(fabs(pattern-actual_class)<=1e-4)    
                {    
                    if(count==0)
                    {
                        continue;
                    }
                    else{
                        count--;
                        break;
                    }
                }
            }
            train_indeces.emplace_back(index);
            test_indeces.erase(find(test_indeces.begin(),test_indeces.end(),index));
            found=true;
        }while(!found);
        if(std::accumulate(train_sizes.begin(),train_sizes.end(),0,[&](int &s,const pair <double,int> &ps) {return s+ps.second;})==0)
        {
            break;
        }
    }

    for(auto &idx:train_indeces)
    {
        train_xpoint_set.emplace_back(this->get_xpointi(idx));
        train_ypoint_set.emplace_back(this->get_ypointi(idx));
    }

    for(auto &idx:test_indeces)
    {
        test_xpoint_set.emplace_back(this->get_xpointi(idx));
        test_ypoint_set.emplace_back(this->get_ypointi(idx));
    }

    Dataset train_dt;
    Dataset test_dt;
    
    train_dt.set_id(this->id+"_train");
    test_dt.set_id(this->id+"_test");

    train_dt.set_category(this->get_category());
    test_dt.set_category(this->get_category());
    
    train_dt.set_data(train_xpoint_set,train_ypoint_set);
    test_dt.set_data(test_xpoint_set,test_ypoint_set);

    train_dt.make_patterns();
    test_dt.make_patterns();
    return pair <Dataset,Dataset>(train_dt,test_dt);
}

pair <Dataset,Dataset> Dataset::train_test_split(double test_size)
{
    vector <Data> train_xpoint_set;
    vector <Data> test_xpoint_set;
    Data train_ypoint_set;
    Data test_ypoint_set;

    mt19937 eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution <int> rand_int(0,this->count()-1);
    vector <int> test_indeces;
    vector <int> train_indeces;

    int test_count=static_cast<int>(this->count()*test_size);

    for(int i=0;i<this->count();i++)
    {
        train_indeces.emplace_back(i);
    }

    bool found;
    int test_idx;
    for(int i=0;i<test_count;i++)
    {
        do{
            test_idx=rand_int(eng);
            found=(find(test_indeces.begin(),test_indeces.end(),test_idx)!=test_indeces.end());
        }while(found);
        test_indeces.emplace_back(test_idx);
        train_indeces.erase(find(train_indeces.begin(),train_indeces.end(),test_idx));
    }

    for(const int &train_idx:train_indeces)
    {
        train_xpoint_set.emplace_back(this->xpoint.at(train_idx));
        train_ypoint_set.emplace_back(this->ypoint.at(train_idx));
    }

    for(const int &test_idx:test_indeces)
    {
        test_xpoint_set.emplace_back(this->xpoint.at(test_idx));
        test_ypoint_set.emplace_back(this->ypoint.at(test_idx));
    }

    Dataset train_dt,test_dt;
    train_dt.set_category(this->get_category());
    test_dt.set_category(this->get_category());

    train_dt.set_id(this->id+"_train");
    test_dt.set_id(this->id+"_test");

    train_dt.set_data(train_xpoint_set,train_ypoint_set);
    test_dt.set_data(test_xpoint_set,test_ypoint_set);

    // if(this->get_category()==Category::CLF)
    // {
    //     train_dt.make_patterns();
    //     test_dt.make_patterns();
    // }

    return make_pair(train_dt,test_dt);
}

ostream &operator<<(ostream &os,Dataset &dataset)
{
    os<<"Id:"<<dataset.id<<endl;
    os<<"Samples:"<<dataset.count()<<endl;
    os<<"Dimension:"<<dataset.dimension()<<endl;
    os<<"Classes:"<<dataset.no_classes()<<endl;
    os<<"=== Standard Deviation ==="<<endl;
    for(int dpos=0;dpos<dataset.dimension();dpos++)
    {
        os<<"Std["<<dpos<<"]:"<<dataset.stdx(dpos)<<endl;
    }
    return os<<endl<<endl;
}


void Dataset::print()
{
    for(int i=0,rows=this->count();i<rows;i++)
    {
        cout<<"R"<<i+1<<":  ";
        for(int j=0,cols=this->dimension();j<cols;j++)
        {
            cout<<this->xpoint[i][j]<<",";
        }
        cout<<this->ypoint[i]<<endl;
    }
}

void Dataset::clean_noise()
{
    vector <int> noisy_dimensions;
    int d=this->dimension();
    for(int i=0;i<d;i++)
    {
        if(count_if(this->xpoint.begin(),this->xpoint.end(),[&](const Data &x) {return fabs(x.at(i)-0)<=1e-4;})==this->count())
        {
            noisy_dimensions.emplace_back(i);
        }    
    }
    
    cout<<"Noisy Dimensions[";
    for(auto &x:noisy_dimensions)
    {
        cout<<x<<" ";
    }
    cout<<"]"<<endl;

    if(noisy_dimensions.empty())
    {
        return;
    }

    vector <Data> new_xpoint;
    for(int i=0,t=this->count();i<t;i++)
    {
        Data new_row;
        for(int j=0,cols=this->dimension();j<cols;j++)
        {
            if(find(noisy_dimensions.begin(),noisy_dimensions.end(),j)!=noisy_dimensions.end())
            {
                continue;
            }
            new_row.emplace_back(this->xpoint[i][j]);           
        }
        new_xpoint.emplace_back(new_row);
    }

    this->set_data(new_xpoint,this->ypoint);
}

void Dataset::statistics()
{
    // Find mean median max min std for each dimension
}
void Dataset::save()
{
    // save the dataset in csv format
    fstream fp;
    fp.open(this->id+".csv",ios::out);
    for(int i=0,rows=this->count();i<rows;i++)
    {
        for(int j=0,cols=this->dimension();j<cols;j++)
        {
            fp<<this->xpoint[i][j]<<",";
        }
        fp<<this->ypoint[i]<<endl;
    }
    fp.close();
}

