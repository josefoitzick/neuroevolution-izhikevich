#ifndef RUN_H
#define RUN_H

#include "all_headers.h"
#include "stdlib.h"

#include <vector>
#include <string>


std::vector<std::string> configNames(std::string directory);
void saveConfig(std::string filename, std::string configName);
void saveRun(Population* population, int n, std::string filename, std::string folder);
void saveResults(std::vector<int> bestFitnes, int n, std::string filename);
float run2(std::string folder, int trial);
float run3(int trial);
float run(int timesPerConfig);


#endif