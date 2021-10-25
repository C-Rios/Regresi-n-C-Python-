#include "exeigennorm.h"
#include "linealregression.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/*En primer lugar se creará una clase "ExEigenNorm", la cual nos permitirá leer un data set,
 * extraer los datos, visualizar los datos, montar la estructura Eigen para normalizar los datos
*/

int main(int argc,char *argv[])
{
    /*Se crea un objeto del tipo exeigennorm, se incluyen los tres argumentos del constructor:
     * Nombre del data set
     * Delimitador
     * Flag (Tiene o no tiene cabecera(header))*/

    ExEigenNorm extraccion(argv[1],argv[2],argv[3]);
    linearRegresion lr;

    /*Se leen los datos del archivo, por la función leerCSV()*/
    std::vector<std::vector<std::string>> dataframe = extraccion.LeerCSV();

    /*Para probar la segunda función CSVtoEigen, se define la cantidad
     * de filas y columnas basados en los datos de entrada*/

    int filas = dataframe.size()+1;
    //std::cout<<"\t\t\t\tFilas de dataframe:"<<filas<<std::endl;
    int columnas = dataframe[0].size();
    //std::cout<<"\t\t\t\tColumnas de dataframe:"<<columnas<<std::endl;

    Eigen::MatrixXd matrizDataFrame = extraccion.CSVtoEigen(dataframe,filas,columnas);


    std::cout<<"\n\n\n\n\t\t\t\t\tData frame \n\n\n"<<matrizDataFrame<<std::endl;

    /* Para desarrollar el primer algoritmo de regresión lineal, en donde se probará con los datos de los vinos (winedata.csv)*/

    Eigen::MatrixXd normMatriz = extraccion.Normalizacion(matrizDataFrame);
    //std::cout<<"\n\n\t\t\t\t\t\t\t\tMatriz normalizada \n\n\n"<<normMatriz<<std::endl;

    //Se desempaca la tupla usando std::tie -> https://en.cppreference.com/w/cpp/utility/tuple/tie

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> splitedData = extraccion.TrainSplitTest(normMatriz,0.8);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    std::tie(X_train,y_train,X_test,y_test) = splitedData;

    /*
    //Inspección visual de la división de los datos para entrenamiento y prueba
    //Variable dependiente
    std::cout <<"Tamaño original "<<normMatriz.rows()<<std::endl;
    std::cout <<"Tamaño entrenamiento filas "<<X_train.rows()<<std::endl;
    std::cout <<"Tamaño entrenamiento columnas "<<X_train.cols()<<std::endl;
    std::cout <<"Tamaño prueba filas "<<X_test.rows()<<std::endl;
    std::cout <<"Tamaño prueba columnas "<<X_test.cols()<<std::endl;
    //Variable independiente
    std::cout <<"Tamaño original "<<normMatriz.rows()<<std::endl;
    std::cout <<"Tamaño entrenamiento filas "<<y_train.rows()<<std::endl;
    std::cout <<"Tamaño entrenamiento columnas "<<y_train.cols()<<std::endl;
    std::cout <<"Tamaño prueba filas "<<y_test.rows()<<std::endl;
    std::cout <<"Tamaño prueba columnas "<<y_test.cols()<<std::endl;
    */

    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());
    /*
        Redimensión de las matrices para ubicación en los vectores ONES (Similar a reshape en numpy)
    */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /*
        Se define el vector theta que le pasará al algoritmo de gradiente descendiente (basicamente un vector
        de ZEROS del mismo tamaño del vector de entrenamiento. Adicional se pasará un alpha y el numero de iteraciones.
    */

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.1;
    int iteraciones = 1000;

    //Se definen las variables de salida que representan los coeficientes y el vector de costo
    Eigen::VectorXd thetaOut;
    std::vector<float> costo;
    std::tuple<Eigen::VectorXd,std::vector<float>> gradiente = lr.GradienteDescendiente(X_train,y_train,theta,alpha,iteraciones);

    std::tie(thetaOut,costo) = gradiente;

    //std::cout<<"Theta ->"<<thetaOut<<std::endl;
/*

    for(auto valor: costo)
    {
        std::cout<<valor<<std::endl;
    }
*/

    //Exportamos a ficheros, costo y thetas

    extraccion.VectorToFile(costo,"costo.txt");
    extraccion.EigenToFile(thetaOut,"thetas.txt");

    return EXIT_SUCCESS;
}
