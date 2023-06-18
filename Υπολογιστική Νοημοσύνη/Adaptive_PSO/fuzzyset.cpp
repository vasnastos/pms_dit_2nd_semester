#include "fuzzyset.hpp"

FuzzySet::FuzzySet(int n)
{
    this->membership.resize(n);
    fill(this->membership.begin(),this->membership.end(),0.0);
}

void FuzzySet::set_membership(int index,double value)
{
    if(index<0 || index>=this->membership.size())
    {
        cerr<<"Index Error:"<<index<<endl;
        return;
    }
    this->membership[index]=value;
}

void FuzzySet::set_rule_base(map <int,int> &rulebase)
{
    this->rule_base=rulebase;
}

int FuzzySet::defuzzify_singleton()
{
    int max_index=0;
    double max_value=0.0;
    for(int i=1,t=this->membership.size();i<t;i++)
    {
        if(this->membership.at(i)>=max_value)
        {
            max_value=this->membership.at(i);
            max_index=i;
        }
    }
    return max_index;
}

int FuzzySet::fuzzy_decision(int &previous_state)
{
    int current_state=this->defuzzify_singleton();
    // if(previous_state==4 && current_state==1)
    // {
    //     return 1;
    // }
    // else if(previous_state==1)
    // {
    //     return 1;
    // }
    // else if(previous_state==2 || previous_state==3)
    // {
    //     return 2;
    // }
    // else
    // {
    for(auto &[cstate,jump_state]:this->rule_base)
    {
        if(cstate==previous_state && jump_state==current_state)
        {
            return current_state;
        }
    }
    // }
    return current_state;
}