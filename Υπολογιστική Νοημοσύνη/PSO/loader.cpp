#include "loader.hpp"
#define P 3.1415

// Function learning problems

ApProblem::ApProblem():Problem(2) {
    Data left_margin={-100,-100};
    Data right_margin={100,100};
    
    this->set_left_margin(left_margin);
    this->set_right_margin(right_margin);
}

double ApProblem::minimize_function(Data &x)
{
    // f(x)=1/4(x1^4)-1/2(x1^2)+1/10(x1)+1/2(x2^2)
    assert(x.size()==this->dimension);
    return ((1.0/4.0)*pow(x[0],4)-(1.0/2.0)*pow(x[0],2)+(1.0/10.0)*x[0]+(1.0/2.0)*pow(x[1],2));
}

Data ApProblem::gradient(Data &x)
{
    Data g;
    g.resize(this->dimension);
    g[0]=pow(x[0],3)-pow(x[0],2)+(1/10);
    g[1]=2*x[1];
    return g;
}

string ApProblem::description()
{
    return "Allufi-Pentiny Problem";
}

Bf1Problem::Bf1Problem():Problem(2) {
    Data left_margin={-50,-50};
    Data right_margin={50,50};
    this->set_left_margin(left_margin);
    this->set_right_margin(right_margin);
}

double Bf1Problem::minimize_function(Data &x)
{
    // f(x)=(x1^2)+2(x2^2)-3/10(cos(4px2))+3/10
    assert(x.size()==this->dimension);
    return pow(x[0],2)+2*pow(x[1],2)-(3/10)*cos(4*P*x[0])-(4/10)*cos(4*P*x[1])+(3/10);
}

Data Bf1Problem::gradient(Data &x)
{
    Data gradient_points;
    gradient_points.resize(this->dimension);
    fill(gradient_points.begin(),gradient_points.end(),0);

    gradient_points[0]=2*x[0]+(6/5)*P*sin(4*P*x[0]);
    gradient_points[1]=4*x[1]+(8/5)*P*sin(4*P*x[1]);

    return gradient_points;
}

string Bf1Problem::description()
{
    return "Bohachevsky-1 Problem";
}

BraninProblem::BraninProblem():Problem(2)  {
    Data left_margin={-5,10};
    Data right_margin={10,15};

    this->set_left_margin(left_margin);
    this->set_right_margin(right_margin);
}

double BraninProblem::minimize_function(Data &x)
{
    // f(x)=(x2-(5.1/(4P^2))*(x1^2)+(5/P)*x1-6)^2+10*(1-(1/8*P))*cos(x1)+10
    return (x[1]-(5.1/(4*pow(P,2)))*pow(x[0],2)+(5/P)*x[0]-6)+10*(1-1/(8*P))*cos(x[0])+10;
}

Data BraninProblem::gradient(Data &x)
{
    Data gradient_points;
    gradient_points.resize(this->dimension);
    //g[0] = -2(x2 - (5.1/(4P^2))*(x1^2) + (5/P)*x1 - 6) * (10.2/(4P^3)*x1 - 5/P)
    gradient_points[0]=2*(x[1]-(5.1/(4*pow(P,2)))*pow(x[0],2)+(5/P)*x[0]-6)*(-2*(5.1/(4*pow(P,2)))*x[0]+(5/P))-2*(x[1]-(5.1/4*pow(P,2))*pow(x[0],2)+(5/P)*x[0]-6) + 10*(1-(1/8*P))*sin(x[0]);
    gradient_points[1]=2*(x[1]-(5.1/(4*pow(P,2)))*pow(x[0],2)+(5/P)*x[0]-6);
    return gradient_points;
}

string BraninProblem::description()
{
    return "Branin Problem";
}