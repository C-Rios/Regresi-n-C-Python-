#include "linealregression.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/*
    Se necesita entrenar el modelo, lo que implica minimizar alguna función de costo
    y de esta forma se puede medir la precisión de la función de hipótesis. La
    función
*/

float linearRegresion::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta)
{
    Eigen::MatrixXd diferencia = pow((X*theta - y).array(),2);
    std::cout<<("xrows ")<<X.rows()<<std::endl;
    return (diferencia.sum()/(2*X.rows()));
}

/*
    Se implementa la función para darle al algoritmo los valores de theta iniciales
    que cambiaran iterativamente hasta que converja al valor mínimo de la función
    de costo. Básicamente describirá el gradiente descendiente: el cual es dado por la
    derivada parcial de la función. La función tiene un alpha que representa el salto
    del gradiente y el número de iteraciones que se necesitan para actualizar theta
    hasta que converja al mínimo esperado
*/

std::tuple<Eigen::VectorXd, std::vector<float>> linearRegresion::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones)
{
    //Almacenamiento temporal de los valores de theta
    Eigen::MatrixXd temporal = theta;
    //Variable con la cantidad de parámetros 'm' (FEATURES)
    int parametros = theta.rows();

    //Variable para ubicar el costo inicial que se actualizará iterativamente con los pesos
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X,y,theta));
    //Para cada iteración se calcula la función de costo
    for(int i = 0; i<iteraciones; i++)
    {
        Eigen::MatrixXd error = X*theta-y;
        for(int j=0; j<parametros;j++)
        {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0)-(alpha/X.rows())*termino.sum();
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X,y,theta));
    }
    return std::make_tuple(theta,costo);
}
