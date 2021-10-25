#define EXEIGENNORM_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>



class ExEigenNorm
{

    //Para el constructor se necesitarán 3 parámetros
    //1. Nombre del dataset
    std::string setDatos;
    //2. Separador de columnas
    std::string delimitador;
    //3. Tiene o no cabecera
    bool header;

public:
    ExEigenNorm(std::string Datos,std::string delimColumnas, bool head):
        setDatos(Datos),
        delimitador(delimColumnas),
        header(head){}
    std::vector<std::vector<std::string>> LeerCSV();

    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int columnas);

    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());

    auto Desviacion(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt());

    Eigen::MatrixXd Normalizacion(Eigen::MatrixXd datos);

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> TrainSplitTest(Eigen::MatrixXd datos, float trainSize);

    void VectorToFile(std::vector<float> vector, std::string nombre);

    void EigenToFile(Eigen::MatrixXd datos, std::string nombre);

};
