#include "solver.hpp"

Solver::Solver()
{
    // load the folder of potential datasets
}

Solver::~Solver()
{

}

void Solver::load(string filename)
{
    this->flush();
    
}

void Solver::flush()
{
    delete this->problem;
    delete this->pso;
}

void Solver::solve()
{

}

Data Solver::get_best_weights()
{

}