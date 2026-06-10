#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "../headers/funciones.h"
#include <numpy/arrayobject.h>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <mutex>
#include <cstdlib> // Para la función rand()
#include <ctime>   // Para la función time()

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;
/*
//vector<Genome> eliminate_less_fit(vector<Genome>)

*/
bool getBooleanWithProbability(double probability) {
    // Generador de números aleatorios
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1); // Distribución uniforme entre 0 y 1

    // Generar un número aleatorio entre 0 y 1
    double randomValue = dis(gen);
     // Comparar el número aleatorio con la probabilidad dada
    return randomValue < probability;
}
bool compareFitness(Genome* a,Genome* b) {
    return a->getFitness() < b->getFitness();
}
bool compareInnovation(Connection& a,Connection& b) {
    return a.getInnovation() < b.getInnovation();
}
bool compareIdNode(Node& a,Node& b) {
    return a.get_id() < b.get_id();
}


// Función para generar un número aleatorio en un rango específico [min, max]
int randomInt(int min, int max) {
    if (min>max){
        std::cout << "Error: min > max" << std::endl;
        return min;
    }else if (min==max){
        std::cout << "Error: min == max" << std::endl;
    }
    
    // Generador de números aleatorios
    std::random_device rd;  // Semilla aleatoria
    std::mt19937 gen(rd()); // Generador Mersenne Twister
    std::uniform_int_distribution<> distrib(min, max-1);

    // Genera un número aleatorio dentro del rango
    return distrib(gen);
}


void deleteDirectory(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string entryName = entry->d_name;
            if (entryName != "." && entryName != "..") {
                std::string fullPath = path + "/" + entryName;
                struct stat entryInfo;
                if (stat(fullPath.c_str(), &entryInfo) == 0) {
                    if (S_ISDIR(entryInfo.st_mode)) {
                        // Es un directorio, eliminar recursivamente
                        deleteDirectory(fullPath);
                    } else {
                        // Es un archivo, eliminar
                        unlink(fullPath.c_str());
                    }
                }
            }
        }
        closedir(dir);
        // Eliminar el directorio ahora que está vacío
        rmdir(path.c_str());
    } else {
        std::cerr << "No se pudo abrir el directorio: " << path << std::endl;
    }
}