#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/run.h"
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

using namespace std;


std::vector<std::string> configNames(std::string directory) {
    // Config names
    vector<string> configNames;
    std::string path = "config";
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "configNames: No se pudo abrir el directorio." << std::endl;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name != "." && name != "..") {
            configNames.push_back(name);
        }
    }

    closedir(dir);
    return configNames;
}
void saveConfig(std::string filename, std::string configName) {
    std::ifstream file(configName); // Abrir el archivo
    std::string line;

    ofstream outfile(filename, ios::app);
    // Leer línea por línea del archivo
    while (std::getline(file, line)) {
        outfile << "  " << line << "\n";
    }
    outfile << "\n";
    outfile.close();
}

void saveRun(Population* population, int n, string filename, string folder) {
    Genome* best = population->getBest();
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "saveRun: No se pudo abrir el archivo." << filename <<endl;
    }
    outfile << "--Results of run:" << n << " --\n";
    outfile << "Best Genome \n";
    outfile << "Genome id: " << best->getId() << "\n";
    outfile << "Genome fitness: " << best->getFitness() << "\n";

    best->sort_connections();
    vector<Connection> connections = best->getConnections();
    int nConnections = static_cast<int>(connections.size());

    outfile << std::setw(5) << "IN"
            << std::setw(5) << "OUT"
            << std::setw(10) << "Weight"
            << std::setw(7) << "Innov" << "\n";

    for (int i = 0; i < nConnections; i++) {
        if (connections[i].getEnabled()) {
            outfile << std::setw(5) << connections[i].getInNode()
                    << std::setw(5) << connections[i].getOutNode()
                    << std::setw(10) << connections[i].getWeight()
                    << std::setw(7) << connections[i].getInnovation() << "\n";
        }
    }
    outfile << "\n";
    outfile.close();

    string file2 = folder + "/best" + to_string(n) + ".txt";
    ofstream outfile2(file2, ios::app);
    if(!outfile2) {
        cerr << "saveRun: No se pudo abrir el archivo outfile2: " << file2 << endl;
    }

    for (int i = 0; i < nConnections; i++) {
        if (connections[i].getEnabled()) {
            outfile2 << connections[i].getInNode() << ";"
                     << connections[i].getOutNode() << ";"
                     << connections[i].getWeight() << "\n";
        }
    }
    outfile2.close();
}

void saveResults(vector<int> bestFitness, int n, string filename) {
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "saveResults: No se pudo abrir el archivo." << filename <<endl;
    }

    outfile << "Summerized results: \n";
    for (int i = 0; i < n; i++){
        outfile << "run: " << i << " bestFitness: " << bestFitness[i] << "\n";
    }
    //Percentage of max fitness
    int max = *max_element(bestFitness.begin(), bestFitness.end());
    int nMax = count(bestFitness.begin(), bestFitness.end(), max);
    float percentage = (nMax * 100.0) / n;
    outfile << "Max fitness: " << max << "\nPercentage: " << percentage << "% (" << nMax << " of " << n << ")\n";
    outfile << "--------------------------------------------------------------\n";
    outfile.close();
}

float run(int timesPerConfig) {

    string filename = "results/results.txt";
    string folder_path_1= "annarchy"; // Ruta de la carpeta que deseas borrar
    string folder_path_2 = "__pycache__"; // Ruta de la carpeta que deseas borrar
    printf("---- Running ----\n");
    vector <string> names = configNames("config");
    int nConfig = static_cast<int>(names.size());
    int evolutions;
    float finalFitness = 0;
    vector <int> bestFitnes;
    // Run Cofigs
    for (int j = 0; j < nConfig; j++){
        printf("---- Config: %s ----\n", names[j].c_str());
        ofstream outfile(filename, ios::app);
        if(!outfile) {
            cerr << "run: No se pudo abrir el archivo." << filename <<endl;
        }
        outfile << "\n---- Results of cofig: " << j << " ----\n";
        outfile.close();
        saveConfig(filename, "config/" + names[j]);
        
        printf("---- Loading Config: %s ----\n", names[j].c_str());
        Parameters parameters("config/" + names[j]);
        printf("---- Loaded Config: %s ----\n", names[j].c_str());
        for (int i = 0; i < timesPerConfig; i++){
            printf("---- Run: %d ----\n", i);
            Population population(&parameters);
            evolutions = parameters.evolutions;

            population.evolution(evolutions, "",0);
            saveRun(&population, i, filename, "");
            bestFitnes.push_back(population.getBest()->getFitness());
            finalFitness += population.getBest()->getFitness();

        }
        saveResults(bestFitnes, timesPerConfig, filename);
        bestFitnes.clear();
    }
    finalFitness = finalFitness / (nConfig*timesPerConfig);
    return finalFitness;
}

float run2(string folder, int trial) {

    string filename = folder + "/results.txt";
    
    int evolutions;
    float finalFitness;
    vector <int> bestFitnes;

    ofstream outfile(filename, ios::app);
    if(!outfile) {
        cerr << "run: No se pudo abrir el archivo." << filename <<endl;
    }

    outfile << "\n---- Results of cofig: ----\n";
    outfile.close();
    saveConfig(filename, folder + "/config.cfg");
    
    Parameters parameters(folder + "/config.cfg");

    Population population(&parameters);
    evolutions = parameters.evolutions;
    population.evolution(evolutions, folder, trial);

    saveRun(&population, 0, filename, folder);
    bestFitnes.push_back(population.getBest()->getFitness());
    finalFitness = population.getBest()->getFitness();

    saveResults(bestFitnes, 1, filename);

    return finalFitness;
}

float run3(int trial) {

    string folder = "results/trial-" + to_string(trial);
    string filename = folder + "/results.txt";
    printf("---- Running ----\n");

    int evolutions;
    float finalFitness;
    vector <int> bestFitnes;

    ofstream outfile(filename, ios::app);
    if(!outfile) {
        cerr << "run: No se pudo abrir el archivo: " << filename <<endl;
    }
    
    printf("---- Loading Config ----\n");
    Parameters parameters("config/config.cfg");
    printf("---- Loaded Config ----\n");


    // Escribir en el archivo config.cfg
    string filename2 = folder + "/config.cfg";
    // Crear la carpeta
    //std::filesystem::create_directories(folder);
    // Crear y abrir el archivo en modo truncado
    ofstream config_file(filename2, ofstream::trunc);
    if (!config_file.is_open()) {
        cerr << "No se pudo abrir el archivo: " << filename2 << endl;
        return 1;
    }
    config_file << "keep=" << parameters.keep << "\n";
    config_file << "threshold=" << parameters.threshold << "\n";
    config_file << "interSpeciesRate=" << parameters.interSpeciesRate << "\n";
    config_file << "noCrossoverOff=" << parameters.noCrossoverOff << "\n";
    config_file << "probabilityWeightMutated=" << parameters.probabilityWeightMutated << "\n";
    config_file << "probabilityAddNodeSmall=" << parameters.probabilityAddNodeSmall << "\n";
    config_file << "probabilityAddLink_small=" << parameters.probabilityAddLinkSmall << "\n";
    config_file << "probabilityAddNodeLarge=" << parameters.probabilityAddNodeLarge << "\n";
    config_file << "probabilityAddLink_Large=" << parameters.probabilityAddLinkLarge << "\n";
    config_file << "largeSize=" << parameters.largeSize << "\n";
    config_file << "c1=" << parameters.c1 << "\n";
    config_file << "c2=" << parameters.c2 << "\n";
    config_file << "c3=" << parameters.c3 << "\n";
    config_file << "numberGenomes=" << parameters.numberGenomes << "\n";
    config_file << "numberInputs=" <<  parameters.numberInputs << "\n";
    config_file << "numberOutputs=" << parameters.numberOutputs << "\n";
    config_file << "evolutions=" << parameters.evolutions << "\n";
    config_file << "n_max=" << parameters.n_max << "\n";
    config_file << "learningRate=" << parameters.learningRate << "\n";
    config_file << "inputWeights=" << parameters.inputWeights[0] << "," << parameters.inputWeights[1] << "\n";
    config_file << "weightsRange=" << parameters.weightsRange[0] << "," << parameters.weightsRange[1] << "\n";
    config_file << "process_max=" << parameters.process_max << "\n";
    config_file << "function=" << parameters.function << "\n";
    config_file << "folder=" << folder << "\n";
    config_file.close();

    outfile << "\n---- Results of cofig: ----\n";
    outfile.close();
    saveConfig(filename, folder + "/config.cfg");

    printf("---- Running NEAT ----\n");
    Population population(&parameters);
    evolutions = parameters.evolutions;
    population.evolution(evolutions, folder, trial);

    saveRun(&population, 0, filename, folder);
    bestFitnes.push_back(population.getBest()->getFitness());
    finalFitness = population.getBest()->getFitness();

    saveResults(bestFitnes, 1, filename);
    
    return finalFitness;
}