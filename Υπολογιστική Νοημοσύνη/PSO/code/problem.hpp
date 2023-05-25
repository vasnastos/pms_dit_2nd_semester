#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <map>
#include <cassert>
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
        mt19937 eng;
    public:
        Problem();
        Problem(int d);
        virtual ~Problem();

        void set_dimension(int dim);
        void set_left_margin(Data &x);
        void set_right_margin(Data &x);
        int get_dimension()const;
        Data get_left_margin()const;
        Data get_right_margin()const;
        bool is_point_in(Data &x);
        double grms(Data &x);

        Data get_sample();
        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
        virtual string description()=0;
        virtual void load(string filepath,int nodes,string wit);
        virtual void flush();
};