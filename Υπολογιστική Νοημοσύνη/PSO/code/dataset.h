#include "problem.h"
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
using namespace std;

enum class Category
{
    CLF,
    REG
};

class Dataset
{
    private:
        string id;
        Category category;
        vector <Data> xpoint;
        Data ypoint;
        Data patterns;
    public:
        Dataset();
        ~Dataset();

        void set_id(string &dataset_id);
        void set_category(Category &cat);

        string get_id()const;
        Category get_category()const;
        string get_named_category()const;

        void read(string filename,string separator);

        Data get_xpointi(int pos);
        double get_ypointi(int pos);

        double xmean(int pos);
        double ymean();
        double xmax(int pos);
        double xmin(int pos);
        double ymax();
        double ymin();
        double stdx(int pos);
        double stdy();

        void normalization(string ntype="min_max");
        void make_patterns();

        int dimension()const;
        int count()const;

        double get_class(double &value);
        double get_class(int &pos);
};