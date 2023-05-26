#include <iostream>
#include <map>
using namespace std;

int main()
{
   int d=320;
   int weight_size=32;
   for(int i=0;i<d;i++)
   {
      cout<<"I:"<<i/weight_size<<"\tJ:"<<i%weight_size<<endl;
   }
   return 0;
}