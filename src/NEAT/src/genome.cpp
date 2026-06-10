#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <fstream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../headers/genome.h"
#include "../headers/funciones.h"
using namespace std;

// ConstructorsGenome::Genome(){}
Genome::Genome(){}
Genome::Genome(int new_id, int num_in, int num_out, Innovation &innov_E, Parameters &parameters_E, int idAnnarchy){
    id = new_id;
    numIn = num_in;
    numOut = num_out;
    fitness = 0;
    innov = &innov_E;
    parameters = &parameters_E;  
    id_annarchy = idAnnarchy;
    adjustedFitness = 0;
    for (int i = 0; i < numIn; i++){
        Node n(i+1, 0);
        nodes.push_back(n);
    }
    for (int i = numIn; i < numIn+numOut; i++){
        Node n(i+1, 2);
        nodes.push_back(n);
    }
    int cInnov;
    for (int i = 0; i < numIn; i++){
        for (int j = numIn; j < numIn+numOut; j++){
            cInnov = innov->addConnection(i+1,j+1);
            
            float minWeight = parameters->weightsRange[0];
            float maxWeight = parameters->weightsRange[1];
            int w = 1;
            int w2 = 1;
            if (minWeight == maxWeight){
                w = rand()%2;
                if (w == 0) w2 = -1; 
            }
            
            //float weight = maxWeight*inh;
            float weight = minWeight + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(maxWeight-minWeight)));
            weight = weight*w2;
            Connection c(i+1, j+1, weight, true, cInnov);
            connections.push_back(c);
        }
    }
    for (int i = 0; i < static_cast<int>(parameters->inputWeights.size()); i++){
        inputWeights.push_back(parameters->inputWeights[i]);
        //printf("Input weight %d: %f\n", i, inputWeights[i]);
    }
    // Verificar el tamaño de inputWeights
    //std::cout << "Tamaño de inputWeights en el constructor: " << inputWeights.size() << std::endl;
}

std::vector<Connection> Genome::getConnections(){ 
    sort_connections();
    return connections;} ;      
std::vector<Node> Genome::getNodes(){ 
    sort_nodes();
    return nodes;}

int Genome::getInNodes(){ return numIn;}

int Genome::getOutNodes(){ return numOut;}
int Genome::getId(){ return id;}
float Genome::getFitness(){ return fitness;}
int Genome::getIdAnnarchy(){ return id_annarchy;}
double Genome::getAdjustedFitness(){ return adjustedFitness;}

Connection* Genome::getConnection(int in_node, int out_node){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInNode() == in_node && connections[i].getOutNode() == out_node){ 
            return &connections[i];
        }
    }
    static Connection null_connection;
    return &null_connection;
}

bool Genome::connectionExist(int in_node, int out_node){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInNode() == in_node && connections[i].getOutNode() == out_node){ 
            return true;
        }
    }
    return false;
}

Connection* Genome::getConnectionId(int innovation){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInnovation() == innovation){
            return &connections[i];
        }
    }
    static Connection null_connection;
    return &null_connection;
}

int Genome::getIndexConnection(int innovation){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInnovation() == innovation){
            return i;
        }
    }
    return -1;
}

Node& Genome::getNode(int id){
    for(int i = 0; i < static_cast<int>(nodes.size()); i++){
        if(nodes.front().get_id() == id){
            return nodes.front();
        }
    }
    static Node null_node;
    return null_node;
}

void Genome::setId(int new_id){ id = new_id;}
void Genome::setFitness(float new_fitness){ fitness = new_fitness;}
void Genome::setConnections(std::vector<Connection> new_connections){ connections = new_connections;}
void Genome::setNodes(std::vector<Node> new_nodes){ nodes = new_nodes;}
void Genome::setParameters(Parameters* new_parameters){ parameters = new_parameters;}
void Genome::setInnovation(Innovation* new_innov){ innov = new_innov;}
void Genome::setIdAnnarchy(int new_id_annarchy){ id_annarchy = new_id_annarchy;}
void Genome::setAdjustedFitness(float new_adjustedFitness){ adjustedFitness = new_adjustedFitness;};
// Mutators

// Change weight, this depends
void Genome::changeWeight(int index, float new_weight){
    connections[index].changeWeight(new_weight);
}

// Create new connection
void Genome::createConnection(int in_node, int out_node, float new_weight){
    int innovation = innov->addConnection(in_node,out_node);
    Connection c(in_node, out_node, new_weight, 1, innovation);
    connections.push_back(c);
}

// Create new node
void Genome::createNode(int index){
    // Find connection and disable
    connections[index].setEnabled(0);
    float new_weight = connections[index].getWeight();
    int in_node = connections[index].getInNode();
    int out_node = connections[index].getOutNode();

    // get last id
    int new_id = innov->addNode(in_node,out_node);
    // Add node
    Node n(new_id, 2);
    nodes.push_back(n);

    // last innovation
    int new_innovation1 = innov->addConnection(in_node,new_id);
    int new_innovation2 = innov->addConnection(new_id,out_node);

    // Add two new connections
    //exh or inh value (1 or -1)
    float inh = (rand() % 2);
    if (inh == 0){
        inh = -1;
    }
    float minWeight = parameters->weightsRange[0];
    float maxWeight = parameters->weightsRange[1];

    float weight = minWeight + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(maxWeight-minWeight)));
    Connection c1(in_node, new_id, inh*weight, 1, new_innovation1);
    Connection c2(new_id, out_node, new_weight, 1, new_innovation2);
    
    connections.push_back(c1);
    connections.push_back(c2);
}

// Print genome
void Genome::printGenome() {
    sort_connections();
    cout << "Genome " << id << " Fitness: " << fitness << endl;
    cout << std::setw(5) << "IN"
         << std::setw(5) << "OUT"
         << std::setw(10) << "Weight"
         << std::setw(7) << "Innov" << endl;

    for (int i = 0; i < static_cast<int>(connections.size()); i++) {
        if (connections[i].getEnabled()) {
            cout << std::setw(5) << connections[i].getInNode()
                 << std::setw(5) << connections[i].getOutNode()
                 << std::setw(10) << connections[i].getWeight()
                 << std::setw(7) << connections[i].getInnovation() << endl;
        }
    }
    cout << "---------------------" << endl;
}


float Genome::singleEvaluation(PyObject *load_module, string folder, int trial){
    //Inicializar varibles necesarias
    int n = static_cast<int>(nodes.size());
    int n_max = parameters->n_max; 
    int numConnections = static_cast<int>(connections.size());
    //Obtener npArray
    double data[n*n];
    for (int i = 0; i < n*n; i++){
        data[i] = 0;
    }
    for (int i = 0; i < numConnections; i++) {
        int in_node = connections[i].getInNode();
        int out_node = connections[i].getOutNode();
        double weight = connections[i].getWeight();
        if (in_node >= 0 && in_node < n && out_node >= 0 && out_node <= n) {
            //int index = (out_node-1) * n + (in_node-1);
            int index = (in_node-1) * n + (out_node-1);
            data[index] = weight;
        }
    }

    _import_array();
    npy_intp dims[2] = {n, n};
    PyObject* numpy_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, data);

    // Convertir el vector inputWeights a un array de NumPy
    if (inputWeights.empty()) {
        std::cerr << "Error: inputWeights está vacío en singleEvaluation." << std::endl;
        return -1;
    }

    // Convertir el vector inputWeights a un array de NumPy
    std::vector<float>& inputWeights2 = inputWeights;
    npy_intp inputWeights_dims[1] = { static_cast<npy_intp>(inputWeights2.size()) };
    PyObject* numpy_inputWeights = PyArray_SimpleNewFromData(1, inputWeights_dims, NPY_FLOAT, inputWeights.data());

    //Llamado a función
    PyObject* func = PyObject_GetAttrString(load_module, "snn");
    PyObject* args = PyTuple_Pack(7, PyFloat_FromDouble(double(parameters->numberInputs)), PyFloat_FromDouble(double(parameters->numberOutputs)), PyFloat_FromDouble(double(n_max)), PyFloat_FromDouble(double(id_annarchy)), numpy_array, numpy_inputWeights, PyFloat_FromDouble(double(trial)));
    PyObject* callfunc = PyObject_CallObject(func, args);
    //Set de fit
    double value = PyFloat_AsDouble(callfunc);
    setFitness(value);


    //Decref de variables necesarias
    Py_DECREF(numpy_array);
    Py_DECREF(args);

    return value;
}

void Genome::mutation(string filenameInfo){
    float add_node, add_link;

    ofstream outfile(filenameInfo, ios::app);
    outfile << " -> Genome id: " << id << " idAnnarchy: " << id_annarchy << endl;

    //Probabilidades
    if (static_cast<int>(nodes.size()) < parameters->largeSize){
        add_node = parameters->probabilityAddNodeLarge;
        add_link = parameters->probabilityAddLinkLarge;
    }else{
        add_node = parameters->probabilityAddNodeSmall;
        add_link = parameters->probabilityAddLinkSmall;
    }

    // mutate weight
    if (getBooleanWithProbability(parameters->probabilityWeightMutated)){
        parameters->mutacionPeso.back() += 1;

        outfile << " ----> Mutate weight --" ;
        int n = connections.size();
        int index =  randomInt(0,n);
        Connection connection = connections[index];

        while (!connection.getEnabled()){
            index =  randomInt(0,n);
            connection = connections[index];
        }
        //Random delta weight between -10 or 10
        //float modification_weight = ((rand() % 200 - 100)/100.0)*parameters->learningRate;
        float modification_weight = parameters->learningRate;
        int exc = rand() % 2;
        if (exc == 1){
            modification_weight = -1*modification_weight;
        }

        changeWeight(index,modification_weight);
        outfile << " index: " << index << " modification_weight: " << modification_weight << endl;
    }else{
        outfile << " ----> Not mutate weight " << endl;
    }

    // add node
    int n_max = parameters->n_max;
    if (getBooleanWithProbability(add_node) && n_max >  (int)(nodes.size())){
        parameters->agregarNodos.back() += 1;

        outfile << " ----> Add node --" << endl;
        int n = connections.size();
        int index = randomInt(0,n);

        while (!(connections[index].getEnabled())){
            index = randomInt(0,n);
        }
        createNode(index);
        outfile << " in: " << connections[index].getInNode() << " out: " << connections[index].getOutNode() << endl;
    }

    // add connection
    if (getBooleanWithProbability(add_link)){
        parameters->agregarLinks.back() += 1;

        outfile << " ----> Add link --" << endl;
        int n = nodes.size();
        int in_node = randomInt(0,n);
        int out_node = randomInt(0,n);

        while (in_node == out_node){
            out_node = randomInt(0,n);
        }
        outfile << " in: " << in_node << " out: " << out_node << endl;

        if (connectionExist(in_node, out_node)){
            Connection* conn = getConnection(in_node, out_node);
            if (!conn->getEnabled()){
                conn->setEnabled(true);
            }
        }else{
            float minWeight = parameters->weightsRange[0];
            float maxWeight = parameters->weightsRange[1];

            float initial_weight = minWeight + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(maxWeight-minWeight)));
            float weight = (rand() % 200 - 100)/100.0;
            weight = weight + initial_weight;

            float inh = randomInt(0,2);            
            if (inh == 1){
                weight = -weight;
            }
            createConnection(in_node, out_node, weight);
        }
    }
    // ¿?
    if (getBooleanWithProbability(parameters->probabilityInputWeightMutated)){
        parameters->mutacionPesoInput.back() += 1;
        outfile << " ----> Mutate input weight --" << endl;
        int n = inputWeights.size();
        int index = randomInt(0,n);
        //Random delta weight between -1 and 1
        float weight = ((rand() % 200 - 100)/100.0)*parameters->learningRate;
        float new_inputWeights = inputWeights[index] + weight;

        inputWeights.at(index) = new_inputWeights;
        //cout << inputWeights[index] << endl;   
    }

    outfile.close();
}

float Genome::compatibility(Genome g1){
    float c1, c2, c3, e, d, n, value;
    vector<Connection> connectionsG1 = getConnections();
    vector<Connection> connectionsG2 = g1.getConnections();
    vector<Node> nodesG1 = getNodes();
    vector<Node> nodesG2 = g1.getNodes();

    int maxConnection, notMaxConnection;
    bool flag;
    if (connectionsG2.back().getInnovation() > connectionsG1.back().getInnovation()){
        maxConnection = connectionsG2.back().getInnovation();
        notMaxConnection = connectionsG1.back().getInnovation();
        flag = true;
    }else{
        maxConnection = connectionsG1.back().getInnovation();
        notMaxConnection = connectionsG1.back().getInnovation();
        flag = false;
    }

    if (nodesG2.size() < 20 && nodesG1.size() < 20){
        n = 1;
    }else{
        if (nodesG2.size() > nodesG1.size()){
            n = nodesG2.size();
        }else{
            n = nodesG1.size();
        }
    }
    
    
    int matching = 0;
    int weightDifference = 0;

    int count1 = 0;
    int count2 = 0;
    d=0;
    e=0;

    for (int i = 0; i < notMaxConnection; i++){
        if (connectionsG2[count1].getInnovation() == i){
            if (connectionsG1[count2].getInnovation() == i){
                matching++;
                weightDifference += abs(connectionsG2[count1].getWeight() - connectionsG1[count2].getWeight());
                count1++;
                count2++;
            }else{
                d++;
                count1++;
            }   
        }else if (connectionsG1[count2].getInnovation() == i){
            d++;
            count2++;
        }
    }
    for (int i = notMaxConnection; i < maxConnection; i++){
        if (flag){
            if (connectionsG2[count1].getInnovation() == i){
                e++;
                count1++;
            }
        }else{
            if (connectionsG1[count2].getInnovation() == i){
                e++;
                count2++;
            }
        }   
    }
    
    c1 = parameters->c1;
    c2 = parameters->c2;
    c3 = parameters->c3;

    value = ((c1*e)/n) + ((c2*d)/n) + c3*((weightDifference)/n);
    
    return value;
}

void Genome::sort_connections(){
    for (int i = 0; i < (int)(connections.size()); i++){
        for (int j = i+1; j < (int)(connections.size()); j++){
            if (connections[i].getInnovation() > connections[j].getInnovation()){
                Connection temp = connections[i];
                connections[i] = connections[j];
                connections[j] = temp;
            }
        }
    }
}

void Genome::sort_nodes(){
    for (int i = 0; i < (int)(nodes.size()); i++){
        for (int j = i+1; j < (int)(nodes.size()); j++){
            if (nodes[i].get_id() > nodes[j].get_id()){
                Node temp = nodes[i];
                nodes[i] = nodes[j];
                nodes[j] = temp;
            }
        }
    }
}