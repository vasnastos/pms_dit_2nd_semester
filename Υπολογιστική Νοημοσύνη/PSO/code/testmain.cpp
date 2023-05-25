#include <iostream>
#include <map>
using namespace std;

int main()
{
   map <double,int> v{{3.5,10},{4.5,9}};
   for(auto &[key,value]:v)
   {
      value--;
   }

   for(auto &[key,value]:v)
   {
      cout<<key<<":"<<value<<endl;
   }
   return 0;
}