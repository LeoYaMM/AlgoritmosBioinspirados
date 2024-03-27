#include <iostream>

using namespace std;

int main(){
    int distancias[5][5] = {
        {0, 5, 2, 4, 1},
        {5, 0, 3, 0, 2},
        {2, 3, 0, 0, 0},
        {4, 0, 0, 0, 5},
        {1, 2, 0, 5, 0}
    };

    int flujos[5][5] = {
        {0, 1, 1, 2, 3},
        {1, 0, 2, 1, 2},
        {1, 2, 0, 1, 2},
        {2, 1, 1, 0, 1},
        {3, 2, 2, 1, 0}
    };

    int solucion1[] = {4, 2, 1, 3, 5};
    int solucion2[] = {1, 2, 3, 5, 4};

    int f1 = 0, f2 = 0;

    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            f1 += flujos[i][j] * distancias[solucion1[i] - 1][solucion1[j] - 1];
            // cout << flujos[i][j] << " * " << distancias[solucion1[i] - 1][solucion1[j] - 1] << " = " << f1 << endl;
            f2 += flujos[i][j] * distancias[solucion2[i] - 1][solucion2[j] - 1];
            // cout << flujos[i][j] << " * " << distancias[solucion2[i] - 1][solucion2[j] - 1] << " = " << f2 << endl;
        }
    }

    cout << "Solucion 1: " << f1 << endl;
    cout << "Solucion 2: " << f2 << endl;

    return 0;
}