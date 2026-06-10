#include "../headers/connection.h"
#include <stdio.h>

using namespace std;

// Constructor de la clase
Connection::Connection(int c_in_node, int c_out_node, float c_weight, bool c_enabled, int c_innovation){
    in = c_in_node;
    out = c_out_node;
    weight = c_weight;
    enabled = c_enabled;
    innovation = c_innovation;
}

Connection::Connection(){
}

// Getters         
int Connection::getInNode() const{
    return in;
}
int Connection::getOutNode() const{
    return out;
}
float Connection::getWeight() const{
    return weight;
}
bool Connection::getEnabled(){
    return enabled;
}
int Connection::getInnovation(){
    return innovation;
}

// Setters 
void Connection::setEnabled(bool new_enabled){
    enabled = new_enabled;
}
void Connection::setWeight(int new_weight){
    weight = new_weight;
}
void Connection::changeEnabled(){
    enabled = !enabled;
}
void Connection::changeWeight(float new_weight){
    weight = weight + new_weight;
}