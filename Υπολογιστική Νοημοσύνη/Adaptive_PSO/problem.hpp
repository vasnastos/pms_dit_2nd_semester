#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <map>
#include <cassert>
#include <numeric>
#include <algorithm>
using namespace std;
using namespace std::chrono;
namespace fs=std::filesystem;

typedef vector <double> Data;

class Problem
{
    protected:
        int dimension;
        Data left_bound;
        Data right_bound;
        mt19937 eng;
        uniform_real_distribution <double> margins;

    public:
        Problem();
        Problem(int d);
        virtual ~Problem();

        void set_dimension(int dim);
        void set_margins(const double &left_margin,const double &right_margin);
        
        void set_left_bound(const double &value);
        void set_right_bound(const double &value);
        void set_left_bound(const Data &data);
        void set_right_bound(const Data &data);

        int get_dimension()const;
        double get_left_margin()const;
        double get_right_margin()const;

        bool is_point_in(Data &x);
        double grms(Data &x);
        Data get_sample();

        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
};