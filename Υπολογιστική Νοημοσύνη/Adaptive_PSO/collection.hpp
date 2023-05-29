#pragma once
#include "problem.hpp"
#include <cmath>

class Collection
{
    private:
        vector <Data> xaxis_data;
        Data yaxis_data;
    public:
        Collection();
        ~Collection();
        void add_point(Data &x,double &y);
        void get_point(int index,Data &x,double &y);
        bool have_graph_minima(Data &x,double &y,double distance);
        void resize_in_fraction(double fraction);
        int size();
        double get_distance(Data &x,Data &y);
        bool is_point_inside(Data &x,double &y);

        void replace_point(int index,Data &x,double &y);
        void get_best_worst_values(double &besty,double &worsty);
};