#ifndef LINEARREGRESION_H
#define LINEARREGRESION_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class linearRegresion
{
public:
    linearRegresion(){}
    float FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones);
};

#endif // LINEARREGRESION_H
