#include "../headers/innovation.h"
#include <vector>
#include <iostream>

Innovation::Innovation(){}
Innovation::Innovation(int in, int out){
    maxNode = 0;
    maxConnection = 0;
    Link link;
    Split split;
    for (int i = 0; i < in; i++){
        maxNode++;
        split = {0,0,maxNode};
        splits.push_back(split);
    }
    for (int i = 0; i < out; i++){
        maxNode++;
        split = {0,0,maxNode};
        splits.push_back(split);
    }
    
    for (int i = 1; i <= in; i++){
        for (int j = in+1; j <= in+out; j++){
            maxConnection++;
            link = {i,j,maxConnection};
            links.push_back(link);
        }
    }
}

int Innovation::addConnection(int in, int out){
    for (int i = 0; i < maxConnection ; i++){
        if (links[i].in == in && links[i].out == out){
            return links[i].id;
        }
    }
    maxConnection++;
    Link link = {in,out,maxConnection};
    links.push_back(link);
    return maxConnection;
}

int Innovation::addNode(int in, int out){
    for (int i = 0; i < maxNode ; i++){
        if (splits[i].in == in && splits[i].out == out){
            return splits[i].id;
        }
    }

    maxNode++;
    Split split = {in,out,maxNode};
    splits.push_back(split);
    return maxNode;
}