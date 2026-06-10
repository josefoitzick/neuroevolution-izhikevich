#ifndef NODE_H
#define NODE_H

class Node{
    public:
        Node();
        Node(int c_id, int c_type);
        int get_id();
        int get_type();
    private:
        int id;
        int type;
};

#endif