#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <string> // Para manejar cadenas de caracteres

class Parameters{
public:
    Parameters();
    Parameters(const std::string& cfgFilename); // Constructor que carga los parámetros desde el archivo cfg
    Parameters(int numberGenomes, int numberInputs, int numberOutputs, 
        float keep=0.9, float threshold=0.01,
        float interSpeciesRate=0.001, float noCrossoverOff=0.25,
        float probabilityWeightMutated=0.8, 
        float probabilityAddNodeSmall=0.9, float probabilityAddLinkSmall=0.9,
        float probabilityAddNodeLarge=0.9, float probabilityAddLinkLarge=0.9,
        float probabilityInputWeightMutated=0.05, int largeSize=20,
        float c1=1.0, float c2=1.0, float c3=0.4);
    int numberGenomes;
    int numberInputs;
    int numberOutputs;
    float keep;
    float threshold;
    float interSpeciesRate;
    float noCrossoverOff;
    float probabilityWeightMutated;
    float probabilityAddNodeSmall;
    float probabilityAddLinkSmall;
    float probabilityAddNodeLarge;
    float probabilityAddLinkLarge;
    float probabilityInputWeightMutated;
    int largeSize;
    float c1;
    float c2;
    float c3;
    int evolutions;
    int process_max;
    int n_max;
    float learningRate;
    std::vector<float> inputWeights;
    float weightsRange[2] = {0.0f, 0.0f};
    std::string function;

    std::vector<int> mutacionPeso;
    std::vector<int> mutacionPesoInput;
    std::vector<int> agregarNodos;
    std::vector<int> agregarLinks;
    std::vector<int> reproducirInter;
    std::vector<int> reproducirIntra;
    std::vector<int> reproducirMuta;

private:
    // Función para cargar los parámetros desde el archivo cfg
    void loadFromCfg(const std::string& filename);
};

#endif
