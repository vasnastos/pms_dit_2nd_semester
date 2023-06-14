#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <map>
#include <cassert>
#include <numeric>
using namespace std;
using namespace std::chrono;
namespace fs=std::filesystem;

typedef vector <double> Data;

class Problem
{
    protected:
        int dimension;
        double neural_network_left_margin;
        double neural_network_right_margin;
        mt19937 eng;
        uniform_real_distribution <double> margins;

    public:
        Problem();
        Problem(int d);
        virtual ~Problem();

        void set_dimension(int dim);
        void set_margins(const double &left_margin,const double &right_margin);
        void set_neural_left_margin(const double &value);
        void set_neural_right_margin(const double &value);
        int get_dimension()const;
        double get_left_margin()const;
        double get_right_margin()const;
        double get_neural_left_margin()const;
        double get_neural_right_margin()const;
        bool is_point_in(Data &x);
        double grms(Data &x);

        Data get_sample();

        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
};