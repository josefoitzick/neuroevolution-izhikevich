#ifndef POPULATION_H
#define POPULATION_H

#include <vector>

#include "genome.h"
#include "innovation.h"
#include "species.h"
#include "funciones.h"
#include "parameters.h"

class Population{
    
  public:  
    // Constructor de la clase
    Population();
    Population(Parameters *param);

    Parameters parameters;
    Innovation innov;
    std::vector<Genome*> genomes;
    std::vector<Species*> species;
    std::vector<int> idForGenomes;
    float threshold;
    void evaluate(std::string folder, int trial);
    void mutations(std::string file);
    void evolution(int n, std::string folder, int trial);
    void print();
    void print_best();

    int maxGenome;

    // Getters
    std::vector<Genome*> getGenomes();
    // Setters

    //
    Genome* findGenome(int id);
    Genome* getBest();
    int findIndexGenome(int id);
    void eliminate(std::string file);
    void reproduce(std::string file);
    void speciation(std::string file);
    void sort_species();
    Genome* crossover(Genome* g1, Genome* g2);
    int get_annarchy_id();
    void offspringsPerSpecies();

  private:
    int nGenomes;
    int nInputs;
    int nOutputs;
    float keep;
};
#endif