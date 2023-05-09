#include "problem.h"
#include <cassert>


class ApProblem:public Problem
{
    public:
        ApProblem();
        double minimize_function(Data &x);
        Data gradient(Data &x);
        string description();
};

class Bf1Problem:public Problem
{
    public:
        Bf1Problem();
        double minimize_function(Data &x);
        Data gradient(Data &x);
        string description();
};

class BraninProblem:public Problem
{
    public:
        BraninProblem();
        double minimize_function(Data &x);
        Data gradient(Data &x);
        string description();
};