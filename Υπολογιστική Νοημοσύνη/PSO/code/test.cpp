#include <iostream>
#include <cmath>
#include <vector>

// Objective function to be optimized
double objectiveFunction(const std::vector<double>& x) {
    // Example: Rosenbrock function
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double term1 = pow((x[i + 1] - pow(x[i], 2.0)), 2.0);
        double term2 = pow((1.0 - x[i]), 2.0);
        sum += 100.0 * term1 + term2;
    }
    return sum;
}

// Gradient of the objective function
std::vector<double> gradient(const std::vector<double>& x) {
    std::vector<double> grad(x.size(), 0.0);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double term1 = 400.0 * pow(x[i], 3.0) - 400.0 * x[i] * x[i + 1] + 2.0 * x[i] - 2.0;
        double term2 = 200.0 * (x[i + 1] - pow(x[i], 2.0));
        grad[i] += term1;
        grad[i + 1] += term2;
    }
    return grad;
}

// BFGS optimization algorithm
std::vector<double> bfgsOptimization(const std::vector<double>& initialGuess, double epsilon, int maxIterations) {
    const double alpha = 1.0;
    const double beta = 0.5;
    const double tolerance = 1e-6;

    std::vector<double> x = initialGuess;
    std::vector<double> H(x.size() * x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        H[i * x.size() + i] = 1.0;  // Initial approximation of the Hessian
    }

    int iteration = 0;
    while (iteration < maxIterations) {
        std::vector<double> grad = gradient(x);
        double normGrad = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            normGrad += pow(grad[i], 2.0);
        }
        normGrad = sqrt(normGrad);
        if (normGrad < epsilon) {
            break;  // Convergence criterion
        }

        std::vector<double> p(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            p[i] = -H[i * x.size() + i] * grad[i];
        }

        // Line search
        double t = 1.0;
        while (objectiveFunction(x + t * p) - objectiveFunction(x) - alpha * t * normGrad >= 0) {
            t *= beta;
        }

        std::vector<double> xNew(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            xNew[i] = x[i] + t * p[i];
        }

        std::vector<double> gradNew = gradient(xNew);
        std::vector<double> y(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = gradNew[i] - grad[i];
        }

        std::vector<double> s(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            s[i] = t * p[i];
        }

        std::vector<double> Hs(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                Hs[i] += H[i * x.size() + j] * s[j];
            }
        }

        double ys = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            ys += y[i] * s[i];
        }

        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                H[i * x.size() + j] += (ys + y[i] * Hs[j]) / pow(normGrad, 2.0) - (Hs[i] * Hs[j]) / ys;
            }
        }

        x = xNew;
        iteration++;
    }

    return x;
}

int main() {
    std::vector<double> initialGuess = { -1.2, 1.0 };
    double epsilon = 1e-6;
    int maxIterations = 100;

    std::vector<double> solution = bfgsOptimization(initialGuess, epsilon, maxIterations);

    std::cout << "Optimized solution: ";
    for (double value : solution) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    double objectiveValue = objectiveFunction(solution);
    std::cout << "Objective value: " << objectiveValue << std::endl;

    return 0;
}