#include "../headers/population.h"
#include <iostream>
#include <atomic>
#include <vector>
#include <array>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <numeric>

using namespace std;

// Declarar un mutex global
#include <mutex>
#include <fcntl.h>

std::mutex mtx;

Population::Population(){}
Population::Population(Parameters *param){
    nGenomes = param->numberGenomes;
    nInputs = param->numberInputs;
    nOutputs = param->numberOutputs;
    maxGenome = nGenomes;
    parameters = *param;
    innov = Innovation(nInputs, nOutputs);
    Genome* g = new Genome(0,nInputs, nOutputs, innov, parameters, 0);
    threshold = parameters.threshold;
    Species* s = new Species(g, threshold);
    species.push_back(s);
    genomes.push_back(g);
    for (int i = 1; i < nGenomes; i++){
        g = new Genome(i,nInputs, nOutputs, innov, parameters, i);
        genomes.push_back(g);
        species[0]->add_genome(g);
    }
    keep = parameters.keep;
}

vector<Genome*> Population::getGenomes(){ return genomes;}

Genome* Population::findGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i]->getId() == id){ return genomes[i];}
    }
    static Genome null_genome; // Genoma nulo
    return &null_genome;
}
int Population::findIndexGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i]->getId() == id){ return i;}
    }
    return 0;
}

void Population::sort_species(){
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        for (int j = i+1; j < static_cast<int>(species.size()); j++){
            if (species[i]->genome->getFitness() < species[j]->genome->getFitness()){
                Species *temp = species[i];
                species[i] = species[j];
                species[j] = temp;
            }
        }
    }
}
void Population::print(){
    //print genomes by species
    for (int i = 0; i < (int)(species.size()); i++){
        std::cout << "Species " << i << endl;
        species[i]->print_genomes();
        std::cout << "----------------" << endl;
    }
}

void Population::eliminate(string filenameInfo){
    int id,index,n,size, id_ann;

    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Eliminate --------\n";
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        size = static_cast<int>(species[i]->genomes.size());
        species[i]->sort_genomes();
        n = static_cast<int>(ceil(size * (1-keep)));
        if (n == size) n -= 1;
        for (int j = 0; j < n; j++){
            id = species[i]->genomes.back()->getId();
            id_ann = species[i]->genomes.back()->getIdAnnarchy();
            index = findIndexGenome(id);
            auto x = genomes.begin() + index;
            genomes.erase(x);
            species[i]->genomes.pop_back();
            idForGenomes.push_back(id_ann);
            outfile << "Genome id: " << id << " idAnnarchy: " << id_ann << "  eliminated" << endl;
        }
    }
    outfile.close();
}

void Population::reproduce(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Reproduce --------\n";
    outfile << " number of species: " << species.size() << endl;
    outfile << " number of genomes: " << genomes.size() << endl;

    Genome *g1, *g2;
    float interSpeciesRate;
    int reproduceInterSpecies, reproduceNonInterSpecies, reproduceMutations, indexS1, indexS2, index;

    offspringsPerSpecies();

    interSpeciesRate = (species.size() > 1) ? parameters.interSpeciesRate : 0;
    
    for (int i = 0; i < static_cast<int>(species.size()); i++){

        //std::cout << "Species " << i << " allocatedOffsprings: " << species[i]->allocatedOffsprings << endl;

        reproduceMutations = 0;
        reproduceInterSpecies = static_cast<int>(ceil(species[i]->allocatedOffsprings) * interSpeciesRate);

        if (species[i]->genomes.size() > 1){
            reproduceNonInterSpecies = species[i]->allocatedOffsprings - reproduceInterSpecies;
        }else{
            reproduceNonInterSpecies = 0;
            reproduceMutations = species[i]->allocatedOffsprings;
        }

        outfile << " -> Species: " << i << endl;
        outfile << " ----> reproduceInterSpecies: " << reproduceInterSpecies << endl;
        outfile << " ----> reproduceNonInterSpecies: " << reproduceNonInterSpecies << endl;
        outfile << " ----> reproduceMutations: " << reproduceMutations << endl;

        outfile << " ----> interSpeciesRate: " ;

        for (int j = 0; j < reproduceInterSpecies; j++){
            parameters.reproducirInter.back() += 1;

            outfile << " " << j+1 << "/" << reproduceInterSpecies << " ";
            indexS1 = i;
            indexS2 = randomInt(0,static_cast<int>(species.size()));
            while(indexS1 == indexS2){
                indexS2 = randomInt(0,static_cast<int>(species.size()));
            } // Revisar
            g1 = species[indexS1]->genomes[randomInt(0,static_cast<int>(species[indexS1]->genomes.size()))];
            g2 = species[indexS2]->genomes[randomInt(0,static_cast<int>(species[indexS2]->genomes.size()))];

            Genome* offspring = crossover(g1,g2);
            genomes.push_back(offspring);
        }

        outfile << endl;
        outfile << " ----> nonInterSpeciesRate: " ;
        for (int j = 0; j < reproduceNonInterSpecies; j++){
            parameters.reproducirIntra.back() += 1;

            outfile << " " << j+1 << "/" << reproduceNonInterSpecies << " ";
            g1 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
            g2 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
            
            while (g1->getId() == g2->getId()){
                g2 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
                if (species[i]->genomes.size() <= 1) {
                    std::cout << "ERROR g1 == g2  species[i]->genomes.size() <= 1" << std::endl;
                } //Revisar
            }

            Genome* offspring = crossover(g1,g2);
            genomes.push_back(offspring);
        }

        outfile << endl;
        outfile << " ----> mutations: " ;
        for (int j = 0; j < reproduceMutations; j++){
            parameters.reproducirMuta.back() += 1;

            outfile << " " << j+1 << "/" << reproduceMutations << " ";
            Genome* offspring = new Genome();
            offspring->setParameters(&parameters);
            offspring->setInnovation(&innov);
            offspring->setId(maxGenome);
            maxGenome++;
            
            index = randomInt(0,static_cast<int>(species[i]->genomes.size()));

            offspring->setFitness(genomes[index]->getFitness());
            offspring->setConnections(genomes[index]->getConnections());
            offspring->setNodes(genomes[index]->getNodes());
            offspring->setIdAnnarchy(get_annarchy_id());
            for (int i = 0; i < static_cast<int>(genomes[index]->inputWeights.size()); i++){
                offspring->inputWeights.push_back(genomes[index]->inputWeights[i]);
            }

            offspring->mutation(filenameInfo);
            
            genomes.push_back(offspring);
        }

        outfile << endl;
    }

    outfile.close();
}

void Population::speciation(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Speciation --------\n";

    int nSpecies = static_cast<int>(species.size());
    vector<int> idGenomesSpecies;
    vector<Genome> genomesSpeciation;
    vector<Genome*> auxGenomes = genomes;
    bool flag;

    this->genomes.clear();

    //Define new representative for each species
    for (int i = 0; i < nSpecies; i++){
        //randomly select a genome from the species
        int index = randomInt(0,static_cast<int>(species[i]->genomes.size()));
        
        species[i]->genome = species[i]->genomes[index];
        species[i]->genomes.clear();
        species[i]->genomes.push_back(species[i]->genome);
        genomes.push_back(species[i]->genome);

        for (int j = 0; j < static_cast<int>(auxGenomes.size()); j++){
            if (auxGenomes[j]->getId() == species[i]->genome->getId()){
                auxGenomes.erase(auxGenomes.begin() + j);
                break;
            }
        }
    }

    int compatibility;
    flag = true;
    //Add genomes to species
    for (int i = 0; i < static_cast<int>(auxGenomes.size()); i++){
        sort_species();
        flag = true;
        for (int j = 0; j < static_cast<int>(species.size()); j++){

            compatibility = (*genomes[i]).compatibility(*species[j]->genome);
            if (compatibility < parameters.threshold){                    
                species[j]->add_genome(auxGenomes[i]);
                genomes.push_back(auxGenomes[i]);
                flag = false;
                break;
            }
        }
        if (flag){
            Species* newSpecies = new Species(auxGenomes[i], parameters.threshold);
            species.push_back(newSpecies);
            genomes.push_back(auxGenomes[i]);
        }
    }

    //eliminar species con 0 genomas
    int i =0;
    while (i < static_cast<int>(species.size())) {
        if (species[i]->genomes.size() == 0) {
            species.erase(species.begin() + i); // Eliminar la especie si no tiene genomas
            // No incrementar i ya que el siguiente elemento se ha movido a la posición i
        } else {
            // Solo incrementar i si no se elimina el elemento actual
            i++;
        }
    }
    
    // Extra para informar
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        outfile << "Species " << i << " size: " << species[i]->genomes.size() << endl;
    }

    outfile.close();
}

void Population::evaluate(std::string folder,int trial) {
    // Importar módulo
    //PyObject* name = PyUnicode_FromString("iris_class");
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);
    Py_DECREF(name);

    // Crear un vector para almacenar los valores de fitness de cada genoma
    std::vector<float> fitness_values(nGenomes, 0.0f);

    // Obtener número máximo de procesos
    long max_processes = parameters.process_max;
    /*long max_processes = sysconf(_SC_CHILD_MAX);
    if (max_processes == -1) {
        std::cerr << "Error al obtener el número máximo de procesos" << std::endl;
        return;
    }*/

    // Dividir los genomas entre los procesos
    int genomes_per_process = nGenomes / max_processes; // Redondeo hacia arriba
    if (genomes_per_process == 0){
        genomes_per_process = 1;
    }
    

    // Crear un vector de pipes para la comunicación con los procesos hijos
    std::vector<std::array<int, 2>> pipes(max_processes);

    // Crear un vector para almacenar los IDs de procesos hijos
    std::vector<pid_t> child_processes;

    for (int i = 0; i < max_processes; ++i) {
        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        if (start >= nGenomes) {
            break;
        }
        
        // Crear un nuevo pipe para la comunicación con el proceso hijo
        if (pipe(pipes[i].data()) == -1) {
            std::cerr << "Error al crear pipe" << std::endl;
            return;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // Código para el proceso hijo
            close(pipes[i][0]); // Cerrar el extremo de lectura del pipe en el hijo

            std::vector<float> child_fitness_values(end - start, 0.0f);

            for (int j = start; j < end; ++j) {
                child_fitness_values[j - start] = genomes[j]->singleEvaluation(load_module, folder, trial);
            }

            // Escribir los valores de fitness al proceso padre
            write(pipes[i][1], child_fitness_values.data(), (end - start) * sizeof(float));
            close(pipes[i][1]); // Cerrar el extremo de escritura del pipe

            exit(0); // Salir del proceso hijo después de ejecutar singleEvaluation
        } else if (pid < 0) {
            std::cerr << "Error al crear proceso hijo" << std::endl;
        } else {
            close(pipes[i][1]); // Cerrar el extremo de escritura del pipe en el padre
            child_processes.push_back(pid);
        }
    }

    // Esperar a que todos los procesos hijos terminen y leer los valores de fitness de los pipes
    for (int i = 0; i < static_cast<int>(child_processes.size()); ++i) {
        waitpid(child_processes[i], nullptr, 0);

        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        read(pipes[i][0], fitness_values.data() + start, (end - start) * sizeof(float));
        close(pipes[i][0]); // Cerrar el extremo de lectura del pipe
    }


    ofstream outfile(folder + "/info.txt", ios::app);
    outfile << "\n-------- Evaluation " << " --------\n";
    // Actualizar los valores de fitness en los genomas correspondientes
    for (int i = 0; i < nGenomes; ++i) {
        genomes[i]->setFitness(fitness_values[i]);
        //std::cout << "Genome id: " << genomes[i]->getId() << " Annarchy id: " << genomes[i]->getIdAnnarchy() << " Fitness: " << genomes[i]->getFitness() << std::endl;
    }
    outfile.close();

    Py_DECREF(load_module);
}


// Crossover
Genome* Population::crossover(Genome* g1, Genome* g2){

    vector<Connection> connections_a, connections_b, connections;

    int count_a=0, count_b = 0;
    Genome* offspring = new Genome(maxGenome, nInputs, nOutputs, innov, parameters, get_annarchy_id());
    maxGenome++;
    vector<Node> offNodes;

    // select best fitness
    if (g1->getFitness() > g2->getFitness()){
        offspring->setFitness(g1->getFitness());
        offNodes = g1->getNodes();
        connections_a = g1->getConnections();
        connections_b = g2->getConnections();

    }else{
        offspring->setFitness(g2->getFitness());
        offNodes = g2->getNodes();
        connections_a = g2->getConnections();
        connections_b = g1->getConnections();

    }
    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();
    
    // Add all connections based on the innovation number from both parents, if they are same add them randomly
    // if they are different, add them in order
    while (count_a < connection_size_a && count_b < connection_size_b){
        if (connections_a[count_a].getInnovation() == connections_b[count_b].getInnovation()){
            if (rand() % 2 == 0){
                connections.push_back(connections_a[count_a]);
            }else{
                connections.push_back(connections_b[count_b]);
            }
            count_a++;
            count_b++;
        }
        else if (connections_a[count_a].getInnovation() < connections_b[count_b].getInnovation()){
            connections.push_back(connections_a[count_a]);
            count_a++;
        }
        else{
            count_b++;
        }
    }

    // Add the remaining connections
    while (count_a < connection_size_a)
    {
        connections.push_back(connections_a[count_a]);
        count_a++;
    }

    offspring->setConnections(connections);
    offspring->setNodes(offNodes);
    
    return offspring;

}

void Population::mutations(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Mutations --------\n";
    outfile.close();
    //mutate
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->sort_genomes();
        for (int j = 1; j < (int)(species[i]->genomes.size()); j++){
            species[i]->genomes[j]->mutation(filenameInfo);
        }
    }
}

void Population::evolution(int n, std::string folder, int trial){
    string filenameInfo = folder + "/info.txt";
    string filenameOperadores = folder + "/operadores.txt";
    Genome *best;

    for (int i = 0; i < n; i++){
        ofstream outfile(filenameInfo, ios::app);
        if (i != 0){
            best = getBest();
            outfile << "\n---- Best Genome ----" << endl;
            outfile << "Genome id: " << best->getId() << endl;
            outfile << "Genome idAnnarchy: " << best->getIdAnnarchy() << endl;
            outfile << "Genome fitness: " << best->getFitness() << endl;
        }
        
        outfile << "\n-------- Generation: " << i << " --------\n";
        outfile.close();

        std::cout << "-------- Generation: " << i << " --------" << std::endl;
        parameters.mutacionPeso.push_back(0);
        parameters.mutacionPesoInput.push_back(0);
        parameters.agregarNodos.push_back(0);
        parameters.agregarLinks.push_back(0);
        parameters.reproducirInter.push_back(0);
        parameters.reproducirIntra.push_back(0);
        parameters.reproducirMuta.push_back(0);

        evaluate(folder, trial);
        eliminate(filenameInfo);
        mutations(filenameInfo);
        reproduce(filenameInfo);
        speciation(filenameInfo);
        fstream outfile2(filenameOperadores, ios::app);
        outfile2 << "\n-------- Resumen operadores Generacion: " << i << " --------\n";
        outfile2 << "---> mutacionPeso: " << parameters.mutacionPeso.back() << endl;
        outfile2 << "---> mutacionPesoInput: " << parameters.mutacionPesoInput.back() << endl;
        outfile2 << "---> agregarNodos: " << parameters.agregarNodos.back() << endl;
        outfile2 << "---> agregarLinks: " << parameters.agregarLinks.back() << endl;
        outfile2 << "---> reproducirInter: " << parameters.reproducirInter.back() << endl;
        outfile2 << "---> reproducirIntra: " << parameters.reproducirIntra.back() << endl;
        outfile2 << "---> reproducirMuta: " << parameters.reproducirMuta.back() << endl;
        outfile2.close();
    }
    evaluate(folder, trial);

    ofstream outfile2(filenameOperadores, ios::app);
    outfile2 << "\n-------- Resumen operadores Total --------\n";
    outfile2 << "---> mutacionPeso: " << std::accumulate(parameters.mutacionPeso.begin(), parameters.mutacionPeso.end(), 0) << endl;
    outfile2 << "---> mutacionPesoInput: " << std::accumulate(parameters.mutacionPesoInput.begin(), parameters.mutacionPesoInput.end(), 0) << endl;
    outfile2 << "---> agregarNodos: " << std::accumulate(parameters.agregarNodos.begin(), parameters.agregarNodos.end(), 0) << endl;
    outfile2 << "---> agregarLinks: " << std::accumulate(parameters.agregarLinks.begin(), parameters.agregarLinks.end(), 0) << endl;
    outfile2 << "---> reproducirInter: " << std::accumulate(parameters.reproducirInter.begin(), parameters.reproducirInter.end(), 0) << endl;
    outfile2 << "---> reproducirIntra: " << std::accumulate(parameters.reproducirIntra.begin(), parameters.reproducirIntra.end(), 0) << endl;
    outfile2 << "---> reproducirMuta: " << std::accumulate(parameters.reproducirMuta.begin(), parameters.reproducirMuta.end(), 0) << endl;
    outfile2.close();
}

void Population::print_best(){
    float bestFitness = 0;
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    genomes[bestIndex]->printGenome();
}

Genome* Population::getBest(){
    float bestFitness = 0;
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    
    return genomes[bestIndex];
}

int Population::get_annarchy_id(){
    int idGenome = idForGenomes.back();
    if (idGenome >= 0){
        idForGenomes.pop_back();
        return idGenome;
    }
    return -1;
}

void Population::offspringsPerSpecies() {
    vector<int> offspringsAlloted(species.size(), 0);
    float totalAverageFitness = 0;
    int sum = 0;
    int genomesSize = static_cast<int>(genomes.size());
    int speciesSize = static_cast<int>(species.size());

    for (int i = 0; i < speciesSize; ++i) {
        species[i]->calculateAdjustedFitness();
        species[i]->calculateAverageFitness();
        totalAverageFitness += species[i]->averageFitness;
    }

    // Asignar descendientes proporcionalmente al fitness promedio ajustado
    for (int i = 0; i < speciesSize; ++i) {
        offspringsAlloted[i] = round((species[i]->averageFitness / totalAverageFitness) * parameters.numberGenomes);
        if (offspringsAlloted[i] < 0) offspringsAlloted[i] = 0;
        sum += offspringsAlloted[i];
    }

    // Ajustar para asegurar que el número total de descendientes es exactamente igual a parameters.numberGenomes
    int difference = (parameters.numberGenomes - genomesSize) - sum;
    while (difference != 0) {
        for (int i = 0; i < speciesSize; ++i) {
            if (difference > 0) {
                offspringsAlloted[i]++;
                difference--;
            }else if (difference < 0 && offspringsAlloted[i] > 0) {
                offspringsAlloted[i]--;
                difference++;
            }
            if (difference == 0) break;
        }
    }

    for (int i = 0; i < speciesSize; ++i) {
        species[i]->allocatedOffsprings = offspringsAlloted[i];
    }
}