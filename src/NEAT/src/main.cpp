#include <iostream>
#include <vector>
#include <string>
#include <sstream> // Para dividir líneas
#include <fstream>  // Asegúrate de incluir este encabezado 
#include <cstdlib>  // para std::atof
#include <filesystem>
#include <sys/stat.h>  // Para mkdir
#include <sys/types.h> // Para mkdir


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "../headers/menu.h"

using namespace std;

int main(int argc, char *argv[]) {

    float fitness;
    if (argc != 14){
        std::string folder, parentFolder, subFolder;
        int trial = 0;
        if (argc == 2){
            //std::cout << argc << " entre trial:" << argv[1] << endl;
            trial = atoi(argv[1]);
            folder = "results/trial-"+to_string(trial);
            parentFolder = "results";
            subFolder = parentFolder + "/trial-"+to_string(trial);
        }else{
            //std::cout << argc << endl;
            folder = "results/trial-0";
            parentFolder = "results";
            subFolder = parentFolder + "/trial-0";
        }
        std::cout << "Folder: " << folder << " trial: " << trial << endl;
        // Crear la carpeta padre si no existe
        if (mkdir(parentFolder.c_str(), 0777) == 0 || errno == EEXIST) {
            std::cout << "Carpeta padre creada o ya existe: " << parentFolder << std::endl;
            // Crear la subcarpeta
            if (mkdir(subFolder.c_str(), 0777) == 0) {
                std::cout << "Subcarpeta creada exitosamente: " << subFolder << std::endl;
            } else {
                std::cout << "Error al crear la subcarpeta o ya existe." << std::endl;
            }
        } else {
            std::cout << "Error al crear la carpeta padre." << std::endl;
        }

        setenv("PYTHONPATH", ".", 1);
        Py_Initialize();
        //menu();
        std::cout << "starting" << endl;
        //float fitness = run(1);
        fitness = run3(trial);
        std::cout << "finalized" << endl;
        Py_Finalize();

    } else{

        // Parametros constantes
        int numberGenomes, numberInputs, numberOutputs, evolutions, n_max, process_max;
        float learningRate, inputWeights_min, inputWeights_max, weightsRange_min, weightsRange_max;
        string function;

        std::ifstream file("config/config.cfg"); // Abrir el archivo
        std::string line;

        // Leer línea por línea del archivo
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            if (std::getline(iss, key, '=')) {
                std::string value;
                if (std::getline(iss, value)) {
                    try {
                        // Asignar valor al parámetro correspondiente
                        if (key == "numberGenomes") numberGenomes = std::stoi(value);
                        else if (key == "numberInputs") numberInputs = std::stoi(value);
                        else if (key == "numberOutputs") numberOutputs = std::stoi(value);
                        else if (key == "evolutions") evolutions = std::stoi(value);
                        else if (key == "process_max") process_max = std::stoi(value);
                        else if (key == "n_max") n_max = std::stoi(value);
                        else if (key == "learningRate") learningRate = std::stof(value);
                        else if (key == "inputWeights") { 
                            std::istringstream rangeStream(value);
                            std::string rangePart;
                            std::getline(rangeStream, rangePart, ',');
                            inputWeights_min = std::stof(rangePart);
                            std::getline(rangeStream, rangePart, ',');
                            inputWeights_max = std::stof(rangePart);
                        }
                        else if (key == "weightsRange") {
                            std::istringstream rangeStream(value);
                            std::string rangePart;
                            std::getline(rangeStream, rangePart, ',');
                            weightsRange_min = std::stof(rangePart);
                            std::getline(rangeStream, rangePart, ',');
                            weightsRange_max = std::stof(rangePart);
                        }else if (key == "function") function = value;
                    }catch (const std::exception& e) {
                        std::cerr << "Error parsing key: " << key << ", value: " << value << ". Exception: " << e.what() << std::endl;
                    }
                }
            }
        }

        float keep=std::atof(argv[1]);
        float threshold=std::atof(argv[2]);
        float probabilityInterSpecies=std::atof(argv[3]);
        float probabilityNoCrossoverOff=std::atof(argv[4]);
        float probabilityWeightMutated=std::atof(argv[5]);
        float probabilityAddNodeSmall=std::atof(argv[6]);
        float probabilityAddLink_small=std::atof(argv[7]);
        float probabilityAddNodeLarge=std::atof(argv[8]);
        float probabilityAddLink_Large=std::atof(argv[9]);
        int largeSize=20;
        float c1=std::atof(argv[10]);
        float c2=std::atof(argv[11]);
        float c3=std::atof(argv[12]);
        int trialNumber=std::atoi(argv[13]);

        string folder = "results/trial-"+std::to_string(trialNumber);
        std::string parentFolder = "results";
        std::string subFolder = parentFolder + "/trial-"+std::to_string(trialNumber);

        // Crear la carpeta padre si no existe
        if (mkdir(parentFolder.c_str(), 0777) == 0 || errno == EEXIST) {
            //std::cout << "Carpeta padre creada o ya existe: " << parentFolder << std::endl;
            // Crear la subcarpeta
            if (mkdir(subFolder.c_str(), 0777) == 0) {
                //std::cout << "Subcarpeta creada exitosamente: " << subFolder << std::endl;
            } else {
                //std::cout << "Error al crear la subcarpeta o ya existe." << std::endl;
            }
        } else {
            //std::cout << "Error al crear la carpeta padre." << std::endl;
        }

        string filename = folder + "/config.cfg";
        // Crear la carpeta
        //std::filesystem::create_directories(folder);
        // Crear y abrir el archivo en modo truncado
        ofstream config_file(filename, ofstream::trunc);
        if (!config_file.is_open()) {
            cerr << "No se pudo abrir el archivo config.cfg para escribir." << endl;
            return 1;
        }

        // Escribir los parámetros
        config_file << "keep=" << keep << "\n";
        config_file << "threshold=" << threshold << "\n";
        config_file << "interSpeciesRate=" << probabilityInterSpecies << "\n";
        config_file << "noCrossoverOff=" << probabilityNoCrossoverOff << "\n";
        config_file << "probabilityWeightMutated=" << probabilityWeightMutated << "\n";
        config_file << "probabilityAddNodeSmall=" << probabilityAddNodeSmall << "\n";
        config_file << "probabilityAddLink_small=" << probabilityAddLink_small << "\n";
        config_file << "probabilityAddNodeLarge=" << probabilityAddNodeLarge << "\n";
        config_file << "probabilityAddLink_Large=" << probabilityAddLink_Large << "\n";
        config_file << "largeSize=" << largeSize << "\n";
        config_file << "c1=" << c1 << "\n";
        config_file << "c2=" << c2 << "\n";
        config_file << "c3=" << c3 << "\n";
        config_file << "numberGenomes=" << numberGenomes << "\n";
        config_file << "numberInputs=" << numberInputs << "\n";
        config_file << "numberOutputs=" << numberOutputs << "\n";
        config_file << "evolutions=" << evolutions << "\n";
        config_file << "n_max=" << n_max << "\n";
        config_file << "learningRate=" << learningRate << "\n";
        config_file << "inputWeights=" << inputWeights_min << "," << inputWeights_max << "\n";
        config_file << "weightsRange=" << weightsRange_min << "," << weightsRange_max << "\n";
        config_file << "process_max=" << process_max << "\n";
        config_file << "function=" << function << "\n";
        config_file << "folder=" << folder << "\n";
        config_file.close();

        setenv("PYTHONPATH", ".", 1);
        Py_Initialize();

        fitness = run2(folder, trialNumber);
        Py_Finalize();
    }
    std::cout << "Final fitness: " << std::endl;
    std::cout << fitness << std::endl;
    return 0;
}