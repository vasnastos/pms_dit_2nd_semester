#pragma once
#include "astring.hpp"
#include "problem.hpp"
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <set>
using namespace std;

enum class Category
{
    CLF,
    REG
};

class Dataset
{
    private:
        Category category;
        vector <Data> xpoint;
        Data ypoint;
        Data patterns;
    public:
        string id;
        Dataset();
        ~Dataset();

        void set_id(string &dataset_id);
        void set_category(Category &cat);
        void set_data(vector <Data> &xpoint_set,Data &ypoint_set);
        void read(string filename,string separator,bool has_categorical=false);

        string get_id()const;
        Category get_category()const;
        string get_named_category()const;
        Data get_xpointi(int pos);
        double get_ypointi(int pos);

        double xmean(int pos);
        double ymean();
        double xmax(int pos);
        double xmin(int pos);
        double ymax();
        double ymin();
        double stdx(int pos);
        double stdy();

        void normalization(string ntype="min_max");
        void make_patterns();

        int dimension()const;
        int count()const;

        double get_class(double &value);
        double get_class(int &pos);
        int no_classes()const;

        pair <Dataset,Dataset> stratify_train_test_split(double test_size=0.3);
};