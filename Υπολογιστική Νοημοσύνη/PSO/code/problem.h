#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
using namespace std;
using namespace std::chrono;
namespace fs=std::filesystem;

typedef vector <double> Data;

class Problem
{
    protected:
        int dimension;
        Data left_margin;
        Data right_margin;
        mt19937 mt;
    public:
        Problem(int d);
        void set_left_margin(Data &x);
        void set_right_margin(Data &x);
        int get_dimension()const;
        Data get_left_margin()const;
        Data get_right_margin()const;
        Data get_sample();
        
        bool is_point_in(Data &x);
        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
        virtual string description()=0;
        double grms(Data &x);
        virtual ~Problem();
};