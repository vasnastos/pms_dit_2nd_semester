#include "problem.h"
#include "dataset.h"

class MlpProblem
{
    private:
        Dataset *d;
        int nodes;
        vector <double> weights;
    public:
        MlpProblem(Dataset *d);
        ~MlpProblem();
};