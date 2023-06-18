#include <iostream>
#include <vector>
#include <map>
#include <numeric>

using namespace std;

class FuzzySet
{   
    public:
        vector <double> membership;
        map <int,int> rule_base;

        FuzzySet(int n);

        void set_membership(int index,double value);
        void set_rule_base(map <int,int> &rulebase);
        int defuzzify_singleton();
        int fuzzy_decision(int &previous_state);
};