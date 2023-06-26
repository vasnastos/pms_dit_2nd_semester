#ifndef PROBLEM_H
#define PROBLEM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
# include <string>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "base.hpp"

using namespace std;
using namespace std::chrono;


typedef vector<double> Data;
typedef vector<Data> Matrix;

/** Gia na kano optimize tin f(x) **/
/**
 * @brief The Problem class, geniki klasi ylopoihshs
 * problimaton beltistopoihshs.
 */
class Problem
{
protected:
    /** dimension = > diastasi synartisis. Gia ta neuronika
     *  einai poses parametrous exei to neuroniko **/
    int dimension;
    /** left=> einai to aristero akro tis synartisis.
     *     Gia ta neuronika diktya einai to aristero akro ton baron.
    **/
    /** right = > einai to dexi akro tis synartisis.
     */
    Data left, right;
    /**
     * @brief bestx => einai to kalytero simeio pou exei ftasei
     * i sunartisi, diladi kapoio topiko elaxisto. Gia ta neuronika
     * diktya einai to kalytero synolo baron.
     */
    Data bestx;
    /**
     * @brief besty => einai i kalyteri timi tis synartisis. Gia ta
     * neuronika diktya tha einai i pio xamili timi stin synartisi sfalmatos.
     */
    double besty;
    int functionCalls;
public:
    /**
     * @brief Problem, synartisi dimioyrgias
     * @param n
     */
    Problem(int n);

    /**
     * @brief getDimension, epistrefei tin diastasi tou problimatos
     * Sta neuronika diktya einai to plithos ton parametron tou
     * diktyou.
     * @return
     */
    int getDimension() const;

    /**
     * @brief getSample, epistrefei ena neo tyxaio sample sto
     * pedio orismou tis synartisis
     * @return
     */
    virtual Data getSample();
    /**
     * @brief setLeftMargin, orizei to aristero akro tou
     * problimatos veltistopoihsis
     * @param x
     */
    void setLeftMargin(Data &x);
    /**
     * @brief setRightMargin, orizei to dexi akro tou
     * problimatos veltistopoihshs
     * @param x
     */
    void setRightMargin(Data &x);
    /**
     * @brief getLeftMargin, epistrefei to aristero
     * akro tou problimatos veltistopoihshs
     * @return
     */
    Data getLeftMargin() const;
    /**
     * @brief getRightMargin, epistrefei to dexi akro
     * tou problimatos veltistopoihshs
     * @return
     */
    Data getRightMargin() const;
    /**
     * @brief funmin, epistrefei tin synartisi f(x)
     *      * Gia ta neuronika tha epistrefei tin synartisi
     * sfalmatos me weights ta x.
     * @param x
     * @return
     */
    virtual double funmin(Data &x) = 0;
    /**
     * @brief gradient, epistrefei tin paragogo tis f(x)
        Sta neuronika diktya epistrefei tin paragogo
        toy sfalmatos.
     * @param x
     * @return
     */
    virtual Data gradient(Data &x) = 0;
    /**
     * @brief statFunmin Kalei prota tin funmin(x)
     * kai diatirei to bestvalue kai kanei update
     * ta function calls
     * @param x
     */
    double statFunmin(Data &x);
    /**
     * @brief grms, epistrefei tin mesi timi tou gradient(x)
     * @param x
     * @return
     */
    double grms(Data &x);
    /**
     *  Epistrefei to kalytero simeio gia tin f(x)
     */
    Data    getBestx() const;
    /**
     * @brief getBesty, epistrefei tin kalyteri f(x)
     * Sta neuronika tha gyrnaei tin kalyteri timi tou sfalmatos
     * @return
     */
    double  getBesty() const;
    int     getFunctionCalls() const;
    ~Problem();

    bool isPointInside(Data &x);
    virtual Category category();
};


#endif // PROBLEM_H
