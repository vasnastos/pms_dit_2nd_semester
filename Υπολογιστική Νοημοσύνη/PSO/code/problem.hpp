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
        mt19937 eng;
        uniform_real_distribution <double> margins;
    public:
        Problem();
        Problem(int d);
        virtual ~Problem();
        
        void set_margins(double &left_margin,double &right_margin);
        pair <double,double> get_margins()const;
        void set_dimension(int dim);
        int get_dimension()const;

        double grms(Data &x);

        Data get_sample();
        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
        virtual string description()=0;
};