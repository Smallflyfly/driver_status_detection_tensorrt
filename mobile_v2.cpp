//
// Created by fangpf on 2021/5/12.
//

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            cerr << "Cuda failure:" << ret << endl; \
            abort(); \
        } \
    } while(0)


using namespace std;
using namespace nvinfer1;

static const Logger gLogger;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 640;
static const int INPUT_H = 480;


map<string, Weights> loadWeight(const string &weightFile) {
    cout << "Loading weight" << endl;
    map<string, Weights> weightMap;

    ifstream f(weightFile);
    assert(f.is_open() && "unable to load weight file");

    int count = 0;
    input >> count;
    assert(count > 0 && "weight file is invalid");

    while (count --) {
        Weights weight{DataType::kFLOAT, nullptr, 0};
        int size;
        string name;
        input >> name >> dec >> size;
        uint32_t *val = reinterpret_cast<uint32_t*>(malloc(uint32_t) * size);
        for (int i = 0; i < size; ++i) {
            input >> hex >> val[i];
        }
        weight.count = size;
        weight.type = DataType::kFLOAT;
        weight.values = val;

        weightMap[name] = weight;
    }

    return weightMap;
}

IScaleLayer *addBN2d(INetworkDefinition *network, ITensor &input, map<string, Weights> weightMap, string layerName, float eps) {
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *bias = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;

    int len = weightMap[layerName + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));

    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
        shval[i] = bias[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        pval[i] = 1.0;
    }

    Weights scale{DataType::kFLOAT, scval, len};
    Weights shift{DataType::kFLOAT, shval, len};
    Weights power{DataType::kFLOAT, pval, len};

    IScaleLayer *scaleLayer = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scaleLayer);

    return scaleLayer;
}

ILayer *convBNReLU(INetworkDefinition *network, map<string, Weights> weightMap, ITensor &input, int outCh, int kSize, int stride, int g, string layerName) {
    int p = (kSize - 1) / 2;
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer *conv = network->addConvolutionNd(input, outCh, DimsHW{kSize, kSize}, weightMap[layerName + ".0.weight"], emptyWt);
    assert(conv);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setNbGroups(g);

    IScaleLayer *bn = addBN2d(network, *conv->getOutput(0), weightMap, layerName + ".1", 1e-5);

    // relu 6
    IActivationLayer *relu1 = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float ) * 1));
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shval[0] = -6.0;
    scval[0] = 1.0;
    pval[0] = 1.0;
    Weights shift{DataType::kFLOAT, shval, 1};
    Weights scale{DataType::kFLOAT, scval, 1};
    Weights power{DataType::kFLOAT, pval, 1};

    IScaleLayer *scaleRelu = network->addScale(*relu1->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power);
    assert(scaleRelu);

    IActivationLayer *relu2 = network->addActivation(*scaleRelu->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IElementWiseLayer *ew = network->addElementWise(*relu1->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUB);
    assert(ew);

    return ew;
}

ILayer *invertRes(INetworkDefinition *network, map<string, Weights> weightMap, ITensor &input, int inCh, int outCh, int stride, int expandRatio, string layerName) {
    int hiddenDim = inCh * expandRatio;
    bool useResConnect = (stride == 1 && inCh == outCh);
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
//    ITensor *tensor = nullptr;
    IScaleLayer *bn = nullptr;
    if (expandRatio != 1) {
        ILayer *conv1 = convBNReLU(network, weightMap, input, hiddenDim, 3, 1, 1, layerName + ".conv.0");
        ILayer *conv2 = convBNReLU(network,weightMap, *conv1->getOutput(0), hiddenDim, 3, stride, hiddenDim, layerName + ".conv.1");
        IConvolutionLayer *conv3 = network->addConvolutionNd(*conv2->getOutput(0), outCh, DimsHW{1, 1}, weightMap[layerName + ".conv.2.weight"], emptyWt);
        assert(conv3);
        *bn = addBN2d(network, *conv3->getOutput(0), weightMap, layerName[layerName + ".conv.3"], 1e-5);
//        tensor = *bn->getOutput(0);

    } else {
        ILayer *conv1 = convBNReLU(network, weightMap, input, hiddenDim, 3, stride, hiddenDim, layerName + ".conv.0");
        IConvolutionLayer *conv2 = network->addConvolutionNd(*conv1->getOutput(0), outCh, DimsHW{1. 1}, weightMap[layerName + "conv.1.weight"], emptyWt);
        assert(conv2);
        *bn = addBN2d(network, *conv2->getOutput(0), weightMap, layerName + ".conv.2");
//        tensor = bn->getOutput(0);
    }
    if (useResConnect) {
        IElementWiseLayer *ew = network->addElementWise(input, *bn->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew);
        return ew;
    }
    return bn;
}

ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config, int maxBatchSize) {
    INetworkDefinition *network = builder->createNetworkV2(0U);
    assert(network != nullptr);
    ITensor *data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims{3 * INPUT_W * INPUT_H});
    assert(data);

    map<string, Weights> weightMap = loadWeight("driver_status_detection_mobile_v2.wts");

    auto conv1 = convBNReLU(network, weightMap, data, 32, 3, 2, 1, "features.0");
    // [1, 16, 1, 1] t c n s
    auto ir1 = invertRes(network, weightMap, *conv1->getOutput(0), 32, 16, 1, 1, "features.1");

    // [6, 24, 2, 2] t c n s
    auto ir2 = invertRes(network, weightMap, *ir1->getOutput(0), 16, 24, 2, 6, "features.2");
    auto ir3 = invertRes(network, weightMap, *ir2->getOutput(0), 24, 24, 1, 6, "features.3");

    // [6, 32, 3, 2] t c n s
    auto ir4 = invertRes(network, weightMap, *ir3->getOutput(0), 24, 32, 2, 6, "features.4");
    auto ir5 = invertRes(network, weightMap, *ir4->getOutput(0), 32, 32, 1, 6, "features.5");
    auto ir6 = invertRes(network, weightMap, *ir5->getOutput(0), 32, 32, 1, 6, "features.6");

    // [6, 64, 4, 2] t c n s
    auto ir7 = invertRes(network, weightMap, *ir6->getOutput(0), 32, 64, 2, 6, "features.7");
    auto ir8 = invertRes(network, weightMap, *ir7->getOutput(0), 64, 64, 1, 6, "features.8");
    auto ir9 = invertRes(network, weightMap, *ir8->getOutput(0), 64, 64, 1, 6, "features.9");
    auto ir10 = invertRes(network, weightMap, *ir9->getOutput(0), 64, 64, 1, 6, "features.10");

    // [6, 96, 3, 1] t c n s
    auto ir11 = invertRes(network, weightMap, *ir10->getOutput(0), 64, 96, 1, 6, "features.11");
    auto ir12 = invertRes(network, weightMap, *ir11->getOutput(0), 96, 96, 1, 6, "features.12");
    auto ir13 = invertRes(network, weightMap, *ir12->getOutput(0), 96, 96, 1, 6, "features.13");

    // [6, 160, 3, 2]
    auto ir14 = invertRes(network, weightMap, *ir13->getOutput(0), 96, 160, 2, 6, "features.14");
    auto ir15 = invertRes(network, weightMap, *ir14->getOutput(0), 160, 160, 1, 6, "features.15");
    auto ir16 = invertRes(network, weightMap, *ir15->getOutput(0), 160, 160, 1, 6, "features.16");

    // [6, 320, 1, 1]
    auto ir17 = invertRes(network, weightMap, *ir16->getOutput(0), 160, 320, 1, 6, "features.17");

    ILayer *conv2 = convBNReLU(network, weightMap, *ir17->getOutput(0), 1280, 1, 1, 1, "features.18");
    IPoolingLayer *pool = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kAVERAGE, DimsHW{15, 20});
    assert(pool);

    IFullyConnectedLayer *fc = network->addFullyConnected(*pool->getOutput(0), 10, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    assert(fc);

    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    cout << "set output name." << endl;
    network->markOutput(*fc->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1<<20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    cout << "engine built." << endl;

    network->destroy();
    for (auto &mem : weightMap) {
        free((void *)mem.second.values);
    }

    return engine;

}


void APIToModel(IHostMemory *modelStream, int maxBatchSize) {
    // create builder
    IBuilder *builder = createInferBuilder(gLogger);

    // build config
    IBuilderConfig * config = builder->createBuilderConfig();

    // engine
    ICudaEngine *engine = createEngine(*builder, *config, maxBatchSize);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
    config->destroy();
}


int main (int argc, char* argv) {
    if (argc == 1) {
        cerr << "Invalid argument" <<  endl;
        return -1;
    }

    if (string(argv[1]) == "-g") {
        // build engine
        IHostMemory *modelStream{nullptr};
        APIToModel(&modelStream, 1);
        assert(modelStream != nullptr);

        ofstream f("driver_status_detection_mobile_v2.engine", ios::binary);
        if (!f) {
            cerr << "engine file error" << endl;
        }
        f.write(reinterpret_cast<const char*>(modelStream->data(), modelStream->size()));

        cout << "engine file generated." << endl;

        modelStream->destroy();

        return 0;

    }
    char *trtModelStream{nullptr};
    size_t size;
    return 0;
}