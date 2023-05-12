#include "problem.h"
#include <fstream>
#include <sstream>
#include <numeric>


using namespace std;

class Dataset
{
    private:
        string id;
        vector <Data> xpoint;
        Data ypoint;
        vector <string> patterns;
    public:
        Dataset();
        ~Dataset();

        void set_id(string &dataset_id);
        string get_id()const;
        
        void read(string filename,string separator);
        int dimension()const;
        int count()const;

        double xmean(int pos);
        double ymean();
        double xmax(int pos);
        double xmin(int pos);
        double ymax();
        double ymin();
        double stdx(int pos);
        double stdy();



        void normalization(string ntype="min_max");
};