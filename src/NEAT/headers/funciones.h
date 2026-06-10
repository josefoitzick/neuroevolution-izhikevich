#ifndef FUNCIONES_H
#define FUNCIONES_H

#include "node.h"
#include "connection.h"
#include "genome.h"
#include <Python.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

//PyObject* create_numpy_array(vector<Connection> connections, int n, PyObject* numpy_array);
//void createSNN(vector <Genome> genomes, int int_in_nodes, int int_out_nodes);
//void single_evaluation(Genome &genome, PyObject *func, int in, int out);
//void evaluate(Population &population);
//void evolution(Population &population, int n);
//void mutations(Population &population);
bool getBooleanWithProbability(double probability);
// Función de comparación personalizada para ordenar objetos MiObjeto por su atributo valor
bool compareFitness(Genome* a,Genome* b);
bool compareInnovation(Connection& a,Connection& b);
bool compareIdNode(Node& a,Node& b);
int randomInt(int min, int max);
void deleteDirectory(const std::string& path);

#endif