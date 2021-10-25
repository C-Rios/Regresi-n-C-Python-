#include "exeigennorm.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/*
 * Primera función: Lectura de ficheros csv
 * Vector de vectores string
 * La idea es leer línea por línea y almacenar en un vector de vectores tipo string*/

std::vector<std::vector<std::string>> ExEigenNorm::LeerCSV()
{
    //Se abre el archivo para lectura
    std::ifstream Archivo(setDatos);
    //Vector de vectores del tipo string que tendrá los datos del data set
    std::vector<std::vector<std::string>> datosString;
    //Se itera a traves de cada línea del dataset, y se divide el contenido dado por el delimitador previsto por el constructor
    std::string linea = "";

    while(getline(Archivo,linea))
    {
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    //Se cierra el fichero
    Archivo.close();

    //Se retorna el vector de vectores de tipo string
    return datosString;

}

/* Se crea la segunda función para guardar el vector de vectores del tipo string
 * a una matriz Eigen. Similar a Pandas(Python) para presentar un dataframe.
*/

Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int columnas)
{
    //Si tiene cabecera, la removemos
    if(header)
    {
        filas-=1;
    }
    /* Se itera sobre filas y columnas para almacenar en la matriz vacía(Tamaño = filas*columnas),
     * que básicamente almacenará string en un vector: Luego lo pasaremos a float para ser manipulados
    */

    Eigen::MatrixXd dfMAtriz(columnas, filas);

    for(int i = 0; i<filas; i++)
    {
        for(int j = 0; j<columnas; j++)
        {
            dfMAtriz(j,i)    = atof(datosString[i][j].c_str());
        }
    }

    //Se transpone la matriz para tener filas x columnas

    return dfMAtriz.transpose();
}

/* A continuación se van a implementar las funciones para la normalización*/

/*En C++ la palabra reservada auto especifíca que el tipo de la variable que se empieza a declarar
 * se deducirá automáticamente de su inicializador y, para las funciones si su tipo de retorno es
 * auto, se evaluará mediante la expresión del tipo de retorno en tiempo de ejecución.
*/

/*
auto ExEigenNorm::Promedio (Eigen::MatrixXd datos)
{
    //Se ingresa como entrada la matriz de datos y regresa el promedio
    return datos.colwise().mean();
}
*/

/* Todavia no se conoce que retorna datos.colwise().mean(), por ello en C++
 * la herencia del tipo de datos no es directa, es decir, no se sabe que tipo de
 * dato debe retornar. Entonces para ello, se declara el tipo en una expresión
 * "declType" con el fin de tener seguridad de que tipo retornará la función
*/

auto ExEigenNorm::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean())
{
    //Se ingresa como entrada la matriz de datos y regresa el promedio
    return datos.colwise().mean();
}

auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt())
{
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt() ;
}

//Normalización Z-score es una estrategia

Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd data){

    //Se hace la diferencia de cada dato y se le resta el promedio de su fila
    Eigen::MatrixXd diferenciaDatoPromedio = data.rowwise() - Promedio(data);

    //Se imprime el promedio
    //std::cout<<"\n\nPromedio-> "<<Promedio(data)<<std::endl;

    //Se normaliza la matriz al dividir la matriz obtenida anteriormente y la desviación de esta misma matriz
    Eigen::MatrixXd matrizNormalizada = diferenciaDatoPromedio.array().rowwise() / Desviacion(diferenciaDatoPromedio);

    //Se imprime la desviación estándar
    //std::cout<<"\n\nDesviación estándar-> "<<Desviacion(diferenciaDatoPromedio)<<std::endl;

    //Se retorna la matriz normalizada
    return matrizNormalizada;
}

/*
    A continuación se hará una función para dividir los datos de entrenamiento y el conjunto de datos de prueba
*/


std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> ExEigenNorm::TrainSplitTest(Eigen::MatrixXd datos, float trainSize)
{
    int filas = datos.rows();
    int filasTrain = round(trainSize*filas);
    int filasTest = filas-filasTrain;
    //Con Eigen se puede especificar un bloque de una matriz, por ejemplo, se pueden seleccionar las filas superiores indicando el número de filas
    Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);
    /*
     * Seleccionadas las filas de entrenamiento superiores, se seleccionan las primeras 11 columnas de izquierda a derecha
     * que representan las variables independientes (Features)
    */
    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);
    std::cout<<"y_train" << y_train;

    Eigen::MatrixXd test = datos.bottomRows(filasTest);
    Eigen::MatrixXd X_test = test.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    //Finalmente se retorna una tupla dada por el conjunto de datos de prueba y de entrenamiento
    return std::make_tuple(X_train,y_train,X_test,y_test);
}

//Se implementan 2 funciones para exportar a ficheros desde vector y eigen

void ExEigenNorm::VectorToFile(std::vector<float> vector, std::string nombre)
{
    std::ofstream fichero(nombre);

    std::ostream_iterator<float> iterador(fichero,"\n");

    std::copy(vector.begin(),vector.end(),iterador);
}

void ExEigenNorm::EigenToFile(Eigen::MatrixXd datos, std::string nombre)
{
    std::ofstream fichero(nombre);
    if(fichero.is_open())
    {
        fichero<<datos<<"\n";
    }
}

