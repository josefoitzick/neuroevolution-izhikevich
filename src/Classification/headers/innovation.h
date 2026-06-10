#ifndef INNOVATION_H
#define INNOVATION_H

#include <vector>
struct Link {
    int in;
    int out;
    int id;
};

struct Split {
    int in;
    int out;
    int id;
};

class Innovation {
private:
    int maxNode;
    int maxConnection;
    std::vector<Link> links;
    std::vector<Split> splits;

public:
    Innovation();
    Innovation(int in, int out);
    int addConnection(int in, int out);
    int addNode(int in, int out);
};

#endif