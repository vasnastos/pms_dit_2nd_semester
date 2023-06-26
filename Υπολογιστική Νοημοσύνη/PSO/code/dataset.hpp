#pragma once
#include "base.hpp"
#include "problem.hpp"
#include <numeric>
#include <algorithm>
#include <set>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;



class Dataset
{
    private:
        Category category;
        vector <Data> xpoint;
        Data ypoint;
    public:
        Data patterns;
        string id;
        Dataset();
        ~Dataset();

        void set_id(const string &dataset_id);
        void set_category(const Category &cat);
        void set_data(vector <Data> &xpoint_set,Data &ypoint_set);
        void read(string filename);

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

        void statistics();
        void save();

        double get_class(double &value);
        double get_class(int &pos);
        int no_classes()const;

        void print();
        void clean_noise();

        pair <Dataset,Dataset> stratify_train_test_split(double test_size=0.3);
        pair <Dataset,Dataset> train_test_split(double test_size=0.3);

        friend ostream &operator<<(ostream &os,Dataset &dataset);
};