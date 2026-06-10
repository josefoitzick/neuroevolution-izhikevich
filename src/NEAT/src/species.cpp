#include "../headers/species.h"

using namespace std;
Species::Species(Genome *genome_init, float new_threshold)
    : genome(genome_init), threshold(new_threshold){
    
    genomes.push_back(genome_init);
}

void Species::add_genome(Genome *genome){
    genomes.push_back(genome);
}

//Sort genomes by fitness in descending order
void Species::sort_genomes(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        for (int j = i+1; j < (int)(genomes.size()); j++){
            if (genomes[i]->getFitness() < genomes[j]->getFitness()){
                Genome *temp = genomes[i];
                genomes[i] = genomes[j];
                genomes[j] = temp;
            }
        }
    }
}

void Species::print(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        cout << "Genome " << genomes[i]->getId() << " fitness: " << genomes[i]->getFitness() << endl;
    }
}   

void Species::print_genomes(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        cout << "Genome " << genomes[i]->getId() << endl;
        genomes[i]->printGenome();
        cout << "---------------------------------------------"<< endl;
    }
}

void Species::calculateAverageFitness(){
    float sumAdjustedFitness = 0;

    for (int i = 0; i < (int)(genomes.size()); i++){
        sumAdjustedFitness += genomes[i]->getAdjustedFitness();
    }
    averageFitness = sumAdjustedFitness / genomes.size();
}   

void Species::calculateAdjustedFitness(){
    double sumDistance,adjustedFitness;
    if (genomes.size() == 1){
        adjustedFitness = genomes[0]->getFitness();
        genomes[0]->setAdjustedFitness(adjustedFitness);
        return;
    }
    for (int i = 0; i < (int)(genomes.size()); i++){
        sumDistance = static_cast<int>(genomes.size()) - 1;
        adjustedFitness = (genomes[i]->getFitness()) / sumDistance;
        genomes[i]->setAdjustedFitness(adjustedFitness);
    }
}