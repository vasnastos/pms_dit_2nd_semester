#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include "astring.hpp"
using namespace std;
namespace fs=std::filesystem;

#ifdef _WIN32
const char sep='\\';
#elif __linux__
#include <climits>
const char sep='/';
#endif


enum class Category
{
    CLF,
    REG
};

struct Instance
{
    Category category;
    string seperator;
    bool has_categorical_label;
    Instance(Category &c,string sp,bool cat_label);
    Instance();
    string get_named_category()const;
};

class Config
{
    public:
        static map <string,Instance> datasetsdb;
        static vector <string> datasets;
        static void datasets_db_config();
        static string get_id(string filename);
        static Category get_category(string file_id);
        static string get_separator(string file_id);
        static bool categorical_label(string file_id);
};

vector <string> split(string &input_str,string &seperator);