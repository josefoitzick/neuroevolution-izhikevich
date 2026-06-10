#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include <iostream> 
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "node.h"
#include "connection.h"
#include "innovation.h"
#include "parameters.h"

// Clase para Genoma
class Genome {
    
    public:  
        Genome();
        Genome(int new_id, int num_in, int num_out, Innovation &innov, Parameters &parameters, int new_id_annarchy);

        std::vector<Connection> getConnections();
        std::vector<Node> getNodes();

        int getInNodes();
        int getOutNodes();
        float getFitness();
        int getId();
        int getIdAnnarchy();
        double getAdjustedFitness();

        Connection* getConnection(int in_node, int out_node);
        bool connectionExist(int in_node, int out_node);
        Connection* getConnectionId(int innovation);

        int getIndexConnection(int innovation);

        Node& getNode(int id);

        void setId(int new_id);
        void setFitness(float new_fitness);
        void setConnections(std::vector<Connection> new_connections);
        void setNodes(std::vector<Node> new_nodes);
        void setParameters(Parameters* new_parameters);
        void setInnovation(Innovation* new_innov);
        void setIdAnnarchy(int new_id_annarchy);
        void setAdjustedFitness(float new_adjustedFitness);
        // Mutators

        // Change weight, this depends
        void changeWeight(int innovation, float new_weight);

        // Create new connection
        void createConnection(int in_node, int out_node, float new_weight);

        // Create new node
        void createNode(int index);

        // Print genome
        void printGenome();

        float singleEvaluation(PyObject *load_module, std::string folder, int trial);

        void mutation(std::string filenameInfo);

        float compatibility(Genome g1);

        void sort_connections();
        void sort_nodes();

        std::vector<float> inputWeights;

    private:
        int id;
        int numIn;
        int numOut;
        float fitness;
        std::vector<Node> nodes; 
        std::vector<Connection> connections;
        Innovation* innov;
        Parameters* parameters;
        int id_annarchy;
        double adjustedFitness;
};

#endif