#include "loader.h"
#include "pso.h"

int main(int argc,char **argv)
{
    if(argc!=2)
    {
        cerr<<"No suitable parameters are passed"<<endl;
        return;
    }

    int param=atoi(argv[1]);
    Problem *problem;
    if(param==1)
    {
        problem=new ApProblem;
    }
    else if(param==2)
    {
        problem=new Bf1Problem;
    }
    else if(param==3)
    {
        problem=new BraninProblem;
    }

    PSO solver(problem,10000,500);
    solver.solve();

    delete problem;

}
