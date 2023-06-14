#include "base.hpp"

Instance::Instance(Category &c,string sp,bool cat_label):category(c),seperator(sp),has_categorical_label(cat_label) {}

Instance::Instance() {}

map <string,Instance> Config::datasetsdb=map <string,Instance>();
vector <string> Config::datasets=vector <string>();

string Instance::get_named_category()const {
    if(this->category==Category::CLF)
    {
        return "Classification";
    }
    else if(this->category==Category::REG)
    {
        return "Regression";
    }
    return "NONAME";
}


void Config::datasets_db_config()
{
    fs::path pth;
    for(const auto &x:{"..","datasets_db.csv"})
    {
        pth.append(x);
    }

    fstream fp;
    fp.open(pth.string(),std::ios::in);

    if(!fp.is_open())
    {
        cerr<<"File did not open properly(datasetsdb)"<<endl;
        return;
    }

    string line,word;
    vector <string> data;
    bool headers=true;
    Category cat;
    bool has_categorical_label;
    string dataset_separator;
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

        if(data.size()!=4) continue;

        if(trim(data[1])=="clf")
        {
            cat=Category::CLF;
        }
        else if(trim(data[1])=="reg")
        {
            cat=Category::REG;
        }


        if(trim(data[2])=="tab")
        {
            dataset_separator="\t";
        }   
        else if(trim(data[2])=="comma")
        {
            dataset_separator=",";
        }

        if(trim(data[3])=="no")
        {
            has_categorical_label=false;
        }
        else 
        {
            has_categorical_label=true;
        }

        Config::datasetsdb[trim(data[0])]=Instance(cat,dataset_separator,has_categorical_label);
    }
    fp.close();


    // // Get datasets from datasets folder
    pth=fs::path();
    for(const string &x:{"..","datasets"})
    {
        pth.append(x);
    }

    for(const auto &entry:fs::directory_iterator(pth.string()))
    {
        Config::datasets.emplace_back(entry.path().string());
    }
}

string Config::get_id(string filename)
{
    vector <string> split_data;
    string word;
    stringstream ss(filename);
    while(getline(ss,word,sep))
    {
        split_data.emplace_back(word);
    }
    
    string extension=split_data.at(split_data.size()-1).substr(split_data.at(split_data.size()-1).find("."),split_data.at(split_data.size()-1).length());
    return replaceString(split_data.at(split_data.size()-1),extension,"");
}

Category Config::get_category(string file_id)
{
    return Config::datasetsdb[file_id].category;
}

string Config::get_separator(string file_id)
{
    return Config::datasetsdb[file_id].seperator;
}

bool Config::categorical_label(string file_id)
{
    if(Config::datasetsdb.find(file_id)!=Config::datasetsdb.end())
    {
        return Config::datasetsdb[file_id].has_categorical_label;
    }
    return false;
}

vector <string> split(string &input_str,string &seperator)
{
    size_t start_pos=0;
    size_t end_pos=input_str.find(seperator);
    vector <string> data;
    while(end_pos!=string::npos)
    {
        data.emplace_back(input_str.substr(start_pos,end_pos));
        start_pos=end_pos+seperator.length();
        end_pos=input_str.find(seperator,start_pos);
    }
    data.emplace_back(input_str.substr(start_pos,end_pos));
    return data;
}
