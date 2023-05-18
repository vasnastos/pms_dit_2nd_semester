#include "dataset.hpp"

Dataset::Dataset():id("") {}
Dataset::~Dataset() {}

void Dataset::set_id(string &dataset_id)
{
    this->id=dataset_id;
}

string Dataset::get_id()const
{
    return this->id;
}

void Dataset::read(string filename,string separator,bool has_categorical)
{
    this->id=replaceString(filename,".csv","");

    fstream fp;
    fp.open(filename,ios::in);
    string line,word,substring;
    vector <string> data;
    size_t start_pos,seperator_pos;

    if(has_categorical)
    {
        vector <string> labels;
        set <string> distinct_labels;
        int index=0;
        while(getline(fp,line))
        {
            data.clear();
            // Split the data
            start_pos=0;
            seperator_pos=line.find(separator);

            while(seperator_pos!=string::npos)
            {
                substring=line.substr(start_pos,seperator_pos-start_pos);
                data.emplace_back(substring);
                start_pos=seperator_pos+substring.length();
                seperator_pos=line.find(separator,start_pos);
            }

            data.emplace_back(line.substr(start_pos));
            
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
            for(auto itr=distinct_labels.begin();itr!=distinct_labels.end();itr++)
            {
                if(label==*itr) break;
                index++;
            }
            this->ypoint.emplace_back(static_cast<double>(index));
        }
    }
    else
    {
        while(getline(fp,line))
        {
            data.clear();
            // Split the data
            start_pos=0;
            seperator_pos=line.find(separator);

            while(seperator_pos!=string::npos)
            {
                substring=line.substr(start_pos,seperator_pos-start_pos);
                data.emplace_back(substring);
                start_pos=seperator_pos+substring.length();
                seperator_pos=line.find(separator,start_pos);
            }

            data.emplace_back(line.substr(start_pos));
            
            Data row;
            for(int i=0,t=data.size()-1;i<t;i++)
            {
                row.emplace_back(stod(data.at(i)));
            }
            this->xpoint.emplace_back(row);
            this->ypoint.emplace_back(data.at(data.size()-1));
        }
    }
    fp.close();
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
    if(pos<0 || pos>=this->xpoint.size())
    {
        cerr<<"Position Error:"<<pos<<endl;
        return -1.0;
    }

    return std::max_element(this->xpoint.begin(),this->xpoint.end(),[&](const Data &d1,const Data &d2) {return d1.at(pos)<d2.at(pos);})->at(pos);
}

double Dataset::xmin(int pos)
{
    if(pos<0 || pos>=this->xpoint.size() || this->xpoint.size()==0)
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

    return *std::max_element(this->ypoint.begin(),this->ypoint.end(),[&](const double &d1,const double &d2) {return d1<d2;});
}
double Dataset::ymin()
{
    if(this->ypoint.empty())
    {
        return -1.0;
    }
    return *std::min_element(this->ypoint.begin(),this->ypoint.end(),[&](const double &d1,const double &d2) {return d1<d2;});
}

double Dataset::stdx(int pos)
{
    if(pos<0 || pos>=this->xpoint.size() || this->xpoint.size()==0)
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
    if(pos<0 || pos>=this->xpoint.size() || this->xpoint.size()==0)
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
        for(int i=0,t=this->dimension();i<t;i++)
        {
            max_data[i]=this->xmax(i);
            min_data[i]=this->xmin(i);
        }

        for(int i=0,rows=this->count();i<rows;i++)
        {
            for(int j=0,cols=this->dimension();j<cols;j++)
            {
                this->xpoint[i][j]=(this->xpoint[i][j]-min_data[j])/(max_data[j]-min_data[j]);
            }
        }
    }
    else if(ntype=="standardization")
    {
        Data mean_data,std_data;
        mean_data.resize(this->dimension());
        std_data.resize(this->dimension());
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
        }
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

void Dataset::set_category(Category &cat)
{
    this->category=cat;
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
            return "Regressinon";
            break;
        default:
            return "No-Category";
            break;
    }
}


double Dataset::get_class(double &value)
{
    if(this->category==Category::CLF)
    {
        for(auto &pattern:this->patterns)
        {
            if(fabs(value-pattern)<=1e-4)
            {
                return pattern;
            }
        }
    }
    return -1.0;
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