#ifndef MLPPROBLEM_H
#define MLPPROBLEM_H
#include "problem.hpp"
# include "dataset.hpp"
#include "adam.hpp"
#include "pso.hpp"
#include "bfgs.hpp"
#define pi 3.14157

# define SMALLVALUES_METHOD "smallValues"
# define XAVIER_METHOD  "xavier"
# define XAVIERNORM_METHOD  "xavierNorm"

class MlpProblem : public Problem
{
private:
    /** weight=> einai oi parametroi tou neuronikou, to dianysma w **/
    Data weight;
    /** data => einai to dataset ekpaideysis **/
    Dataset *data;
    /** the initialization method for the weights, used in the getSample() **/
    string initMethod;
public:
    string save_distribution_path;
    /**
     * @brief MlpProblem, i synartisi dimioyrgias
     * @param n => posa hidden nodes (vari) exo.
     * @param d => to train set
     */
    MlpProblem(Dataset *d,int n);
    /**
     * @brief setWeights => allazei ton pinaka weight se w.
     * @param w
     */
    /**
     * @brief setInitMethod, allazei tin methodo arxikopoiisis
     * @param m
     */
    void    setInitMethod(string m);
    /**
     * @brief getInitMethod, epistrefei tin methodo arxikopoihshs
     * @return
     */
    string  getInitMethod() const;
    /**
     * @brief getSample, paragei ena neo sample gia ta weights
     * me basi tin methodo arxikopoisis
     * @return
     */
    virtual Data getSample();

    void    setWeights(Data &w);
    /**
     * @brief funmin => einai to sfalma ekpaideysis gia ta weight = x
     * @param x
     * @return
     */
    double  funmin(Data &x);
    /**
     * @brief gradient => einai i paragogos tis synartisis sfalmatos
     * os pros tis parametrous x.
     * @param x
     * @return
     */
    Data    gradient(Data &x);
    /**
     * @brief sig, einai i sigmoidis synartisi gia eisodo x.
     * @param x
     * @return
     */
    double  sig(double x);
    /**
     * @brief sigder, einai i paragogos tis sigmoeidous gia eisodo x.
     * @param x
     * @return
     */
    double  sigder(double x);
    /**
     * @brief getOutput=> einai i timi tou neuronikou gia to protypo x
     * gia weight = x.
     * @param x
     * @return
     */
    double  getOutput(Data &x);
    /**
     * @brief getDerivative=> einai i paragogos gia to protypo x.
     * @param x
     * @return
     */
    Data    getDerivative(Data &x);
    /**
     * @brief getTrainError=> einai to sfalma ekpaideusis,
     * diladi to sfalma sto train set.
     * @return
     */
    double  getTrainError() ;
    /**
     * @brief getTestError=> einai to sfalma sto test set.
     * @param test
     * @return
     */
    double  getTestError(Dataset *test) ;
    /**
     * @brief getClassTestError => einai to classification sfalma
     * gia to test set.
     * @param test
     * @return
     */
    double  getClassTestError(Dataset *test) ;
    void optimize(string optimizer);
    Category category();

    ~MlpProblem();
};

#endif // MLPPROBLEM_H
