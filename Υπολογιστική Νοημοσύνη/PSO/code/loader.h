#include "problem.h"

class CsvLoader:public Problem
{
    private:
        vector <Data> xpoint;
        Data ypoint;
    public:
        CsvLoader(string filepath);
        ~CsvLoader();


};