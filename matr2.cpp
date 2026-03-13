#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <string>

using namespace std;
using namespace chrono;

void createFiles(int size) {
    string nameA = "A_" + to_string(size) + ".txt";
    string nameB = "B_" + to_string(size) + ".txt";

    ifstream testA(nameA);
    ifstream testB(nameB);
    if (testA.good() && testB.good()) return;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1, 1000000);

    ofstream fileA(nameA);
    fileA << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fileA << dis(gen) << " ";
        }
        fileA << endl;
    }
    fileA.close();

    ofstream fileB(nameB);
    fileB << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fileB << dis(gen) << " ";
        }
        fileB << endl;
    }
    fileB.close();
}

vector<vector<double>> readMat(const string& name, int expectedSize) {
    ifstream file(name);
    int n;
    file >> n;
    if (n != expectedSize) {
        cerr << "Error: size mismatch" << endl;
        exit(1);
    }
    vector<vector<double>> M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> M[i][j];
    return M;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<double>> mult(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int threads, bool parallel) {
    int n = (int)A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));

    if (parallel) {
        omp_set_num_threads(threads);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
    }
    else {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
    }
    return C;
}

int main(int argc, char* argv[]) {
    int threads = 1;
    if (argc > 1) {
        threads = atoi(argv[1]);
    }

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };

    for (int s : sizes) {
        createFiles(s);
    }

    double total_check = 0;

    for (int s : sizes) {
        vector<vector<double>> A = readMat("A_" + to_string(s) + ".txt", s);
        vector<vector<double>> B = readMat("B_" + to_string(s) + ".txt", s);

        auto start = high_resolution_clock::now();
        vector<vector<double>> C = mult(A, B, threads, true);
        auto end = high_resolution_clock::now();

        double time = duration<double>(end - start).count();

        string outFile = "C_" + to_string(s) + "_" + to_string(threads) + ".txt";
        writeMatrix(outFile, C);

        cout << "Size: " << s << ", Threads: " << threads
            << ", Time: " << time << "s" << endl;

        for (int i = 0; i < min(5, s); i++) {
            total_check += C[i][i];
        }
    }

    cout << "Check sum: " << total_check << " (prevents optimization)" << endl;
    return 0;
}