#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <map>
#include <cassert>
#include <numeric>
#include <fstream>
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

        double neural_network_left_margin;
        double neural_network_right_margin;

        int function_calls;
        Data best_xpoint;
        double best_ypoint;

    public:
        Problem();
        Problem(int d);
        virtual ~Problem();
        
        void set_margins(double &left_margin,double &right_margin);
        void set_dimension(int dim);
        void set_neural_left_margin(double new_left_margin);
        void set_neural_right_margin(double new_right_margin);
        double get_neural_left_margin()const;
        double get_neural_right_margin()const;


        int get_dimension()const;
        pair <double,double> get_margins()const;
        bool is_point_in(Data &x);
        double grms(Data &x);

        Data get_sample();
        void stat_minimize_function(Data &x);
        virtual double minimize_function(Data &x)=0;
        virtual Data gradient(Data &x) = 0;
        virtual string description()=0;
};