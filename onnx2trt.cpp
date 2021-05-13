//
// Created by fangpf on 2021/5/13.
//

#include "NvInfer.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include "NvOnnxParser.h"


using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

static const char* trtName = "driver_status_detection_mobile_v2.trt";

static Logger gLogger;

void transform(const string &onnxFile) {
    // load onnx file
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(0U);
    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnxFile.c_str(), static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        cerr << "parser onnx file error: " << endl;
        cout << parser->getError(i)->desc() << endl;
    }
    cout << "parser onnx file successfully!" << endl;

    // build engine
    int maxBatchSize = 1;
    builder->setMaxBatchSize(maxBatchSize);
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine * engine = builder->buildEngineWithConfig(*network, *config);

    // serialize model
    IHostMemory *trtModelStream = engine->serialize();
    assert(trtModelStream != nullptr);

    ofstream p(trtName, ios::binary);
    if (!p) {
        cerr << "generate trt file error!" << endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());

    trtModelStream->destroy();
}

int main(int argc, char **argv) {
    // argument -g xxx.onnx
    if (argc != 3) {
        cerr << "argument invalid!" << endl;
    }
    assert(string(argv[1]) == "-g" && "argument error!");
    string onnxFile = string(argv[2]);
    transform(onnxFile);
}