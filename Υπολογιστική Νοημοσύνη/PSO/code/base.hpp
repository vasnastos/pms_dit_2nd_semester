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
#define sep='\\';
#elif __linux__
#include <climits>
#define sep '/'
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
    Instance(Category &c,string sp);
    Instance();
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
};