#include "base.hpp"

Instance::Instance(Category &c,string sp):category(c),separator(sp) {}

map <string,Instance> Config::datasetsdb=map <string,Instance>();
vector <string> Config::datasets=vector <string>();

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

        if(data.size()!=2) continue;

        Category cat;
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
            dataset_separator==",";
        }

        Config::datasetsdb[data[0]]=Instance(cat,dataset_separator);
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
    return Config::datasetsdb[file_id].separator;
}
