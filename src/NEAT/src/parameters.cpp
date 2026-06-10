#include "../headers/parameters.h"
#include <fstream> // Para leer archivos
#include <sstream> // Para dividir líneas
#include <string> // Para manejar cadenas de caracteres
#include <iostream>

// Función para cargar los parámetros desde el archivo cfg
void Parameters::loadFromCfg(const std::string& filename) {
    std::ifstream file(filename); // Abrir el archivo
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
                    if (key == "keep") keep = std::stof(value);
                    else if (key == "threshold") threshold = std::stof(value);
                    else if (key == "interSpeciesRate") interSpeciesRate = std::stof(value);
                    else if (key == "noCrossoverOff") noCrossoverOff = std::stof(value);
                    else if (key == "probabilityWeightMutated") probabilityWeightMutated = std::stof(value);
                    else if (key == "probabilityAddNodeSmall") probabilityAddNodeSmall = std::stof(value);
                    else if (key == "probabilityAddLink_small") probabilityAddLinkSmall = std::stof(value);
                    else if (key == "probabilityAddNodeLarge") probabilityAddNodeLarge = std::stof(value);
                    else if (key == "probabilityAddLink_Large") probabilityAddLinkLarge = std::stof(value);
                    else if (key == "probabilityInputWeightMutated") probabilityInputWeightMutated = std::stof(value);
                    else if (key == "largeSize") largeSize = std::stoi(value);
                    else if (key == "c1") c1 = std::stof(value);
                    else if (key == "c2") c2 = std::stof(value);
                    else if (key == "c3") c3 = std::stof(value);
                    else if (key == "numberGenomes") numberGenomes = std::stoi(value);
                    else if (key == "numberInputs") numberInputs = std::stoi(value);
                    else if (key == "numberOutputs") numberOutputs = std::stoi(value);
                    else if (key == "evolutions") evolutions = std::stoi(value);
                    else if (key == "process_max") process_max = std::stoi(value);
                    else if (key == "n_max") n_max = std::stoi(value);
                    else if (key == "learningRate") learningRate = std::stof(value);
                    else if (key == "inputWeights") { 
                        inputWeights.clear();
                        std::istringstream weightsStream(value);
                        std::string weight;
                        while (std::getline(weightsStream, weight, ',')) {
                            weight.erase(0, weight.find_first_not_of(" \t\n\r\f\v"));
                            weight.erase(weight.find_last_not_of(" \t\n\r\f\v") + 1);
                            if (!weight.empty()) {
                                inputWeights.push_back(std::stof(weight));
                            }
                        }
                    }
                    else if (key == "weightsRange") {
                        std::istringstream rangeStream(value);
                        std::string rangePart;
                        std::getline(rangeStream, rangePart, ',');
                        float minWeight = std::stof(rangePart);
                        std::getline(rangeStream, rangePart, ',');
                        float maxWeight = std::stof(rangePart);
                        weightsRange[0] = minWeight;
                        weightsRange[1] = maxWeight;
                    }else if (key == "function") function = value;
                }catch (const std::exception& e) {
                    std::cerr << "Error parsing key: " << key << ", value: " << value << ". Exception: " << e.what() << std::endl;
                }
            }
        }
    }
}

// Constructor que carga los parámetros desde el archivo cfg
Parameters::Parameters(const std::string& cfgFilename) {
    // Cargar parámetros desde el archivo cfg
    loadFromCfg(cfgFilename);
}

// Constructor por defecto
Parameters::Parameters() {}

// Constructor con parámetros
Parameters::Parameters(int numberGenomes, int numberInputs, int numberOutputs, float keep, float threshold,
            float interSpeciesRate, float noCrossoverOff, float probabilityWeightMutated, 
            float probabilityAddNodeSmall, float probabilityAddLinkSmall, float probabilityAddNodeLarge, float probabilityAddLinkLarge,
            float probabilityInputWeightMutated, int largeSize, float c1, float c2, float c3)
    :numberGenomes(numberGenomes),numberInputs(numberInputs),numberOutputs(numberOutputs),keep(keep),threshold(threshold),
    interSpeciesRate(interSpeciesRate),noCrossoverOff(noCrossoverOff),
    probabilityWeightMutated(probabilityWeightMutated),probabilityAddNodeSmall(probabilityAddNodeSmall),
    probabilityAddLinkSmall(probabilityAddLinkSmall),probabilityAddNodeLarge(probabilityAddNodeLarge),
    probabilityAddLinkLarge(probabilityAddLinkLarge), probabilityInputWeightMutated(probabilityInputWeightMutated),
    largeSize(largeSize),c1(c1),c2(c2),c3(c3){}
