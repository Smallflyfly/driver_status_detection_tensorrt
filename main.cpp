//
// Created by fangpf on 2021/5/8.
//

#include <iostream>
#include <string>
#include "NvInfer.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include <cmath>
#include "cuda_runtime_api.h"


using namespace std;
using namespace nvinfer1;


#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)


int main(int argc, char** argv){
    if (argc !=2){
        cerr << "arguments are invalid" <<  endl;
    }

    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1])== "s") {
        IHostMemory *modelStream{nullptr};
    }
}