#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/menu.h"
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>

using namespace std;

void menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, innovation, innov_c;
  char option;

  Parameters param("config/config-0.cfg");
  // Crear poblacion
  Population p(&param);

  do {
    cout << "Choose an option:  a. create_connection  b. create_node "
        << "c. change weight d. print genome e. exit f. print population"
        <<" g. create snn h. evolucionar i. mutar z. crossover"
        <<" y. print fitness k. print best genome" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      int genome_id;
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;

      //Verificar si el genoma existe
      if (p.findGenome(genome_id) == nullptr) {
        cout << "Genome not found" << endl;
        break;
      }else{
        //Ingresar datos de la conexion
        cout << "Enter in_node: ";
        cin >> in_node;
        cout << "Enter out_node: ";
        cin >> out_node;
        cout << "Enter new_weight: ";
        cin >> new_weight;
        p.findGenome(genome_id)->createConnection(in_node, out_node, new_weight);
        break;
      }
    case 'b':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;

      //Verificar si el genoma existe
      if (p.findGenome(genome_id) == nullptr) {
        cout << "Genome not found" << endl;
        break;
      }else{
        //select connection
        cout << "Enter innovation: ";
        cin >> innov_c;
        int index = p.findGenome(genome_id)->getIndexConnection(innov_c);
        p.findGenome(genome_id)->createNode(index);
        break;
      }
    case 'c':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;

      //Verificar si el genoma existe
      if (p.findGenome(genome_id) == nullptr) {
        cout << "Genome not found" << endl;
        break;
      }else{

        cout << "Enter innovation: ";
        cin >> innovation;
        cout << "Enter new_weight: ";
        cin >> new_weight;
        p.findGenome(genome_id)->changeWeight(innovation, new_weight);
        break;
      }
    case 'd':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;

      //Verificar si el genoma existe
      if (p.findGenome(genome_id) == nullptr) {
        cout << "Genome not found" << endl;
        break;
      }else{
        p.findGenome(genome_id)->printGenome();
        break;
      }
    case 'f':
      //print all genomes
      p.print();
      break;
    case 'g':
      //create snn 
      cout << "Creando " << endl;
      p.evaluate("",0);
      cout << "---------------------------------------------"<< endl;
      break;
    case 'h':
      //evolutionate
      int n;
      cout << "Enter number of evolutions: ";
      cin >> n;
      cout << "Evolucionando " << endl;
      p.evolution(n,"",0);
      cout << "---------------------------------------------"<< endl;
      break;
    case 'i':
      //evolutionate
      cout << "Mutate " << endl;
      //p.mutations();
      cout << "---------------------------------------------"<< endl;
      break;
    case 'j':
      //evolutionate
      cout << "Eliminate " << endl;
      //p.eliminate();
      cout << "---------------------------------------------"<< endl;
      break;
    case 'y':
      //Print fitness for each genome
      for (int i = 0; i < (int)(p.genomes.size()); i++) {
        cout << "Genome " << p.genomes[i]->getId() << " fitness: " << p.genomes[i]->getFitness() << endl;
      }
      break;
    case 'z':
      //Select two genomes
      int genome1, genome2;
      cout << "Enter first genome id: ";
      cin >> genome1;
      cout << "Enter second genome id: ";
      cin >> genome2;

      //Verificar si los genomas existen
      if (p.findGenome(genome1) == nullptr || p.findGenome(genome2) == nullptr) {
        cout << "Genomes not found" << endl;
        break;
      }else{
        Genome* g3 = p.crossover(p.findGenome(genome1), p.findGenome(genome2));
        p.maxGenome++;
        g3->printGenome();
        //Add new genome
        p.genomes.push_back(g3);
        break;
      }
    case 'k':
      //Print best genome
      p.print_best();
      break;
    }
  } while (option != 'e');
}