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

static Logger gLogger;

#define INPUT_BLOB_NAME "input"
#define OUTPUT_BLOB_NAME "output"
#define INPUT_W 640
#define INPUT_H 480


#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)


map<string, Weights> loadWeight(const string& weightFile) {
    cout << "loading weights" << endl;
    map<string, Weights> weightMap;

    ifstream input(weightFile);
    assert(input.is_open() && "unable to open weight file");

    int32_t count;
    input >> count;
    assert(count > 0 && "invalid weight map file");

    while (count --) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t  size;

        string name;
        input >> name >> dec >> size;
        wt.type = DataType::kFLOAT;
        wt.count = size;
        // val
        uint32_t  *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t i = 0; i < size; i++) {
            input >> hex >>  val[i];
        }
        wt.values = val;

        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer *addBN(INetworkDefinition *network, map<string, Weights> weightMap, ITensor &input, const string layerName, float eps) {
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *bias = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;

    int len = weightMap[layerName + ".running_var"].count;
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float ) * len));
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));

    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
        shval[i] = bias[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        pval[i] = 1.0;
    }
    Weights scale{DataType::kFLOAT, scval, len};
    Weights shift{DataType::kFLOAT, shval, len};
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;

    IScaleLayer *bn = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn);

    return bn;
}

ILayer *addHardWish(INetworkDefinition *network, ITensor &input) {
    IActivationLayer *hardSigmoid = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hardSigmoid);
    hardSigmoid->setAlpha(1.0 / 6.0);
    hardSigmoid->setBeta(0.5);
    ILayer *ew = network->addElementWise(input, *hardSigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);

    return ew;
}

ILayer *seLayer(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int expandCh, int w, string layerName) {
    int h = w;
    IPoolingLayer *pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{h, w});
    assert(pool);
    pool->setStrideNd(DimsHW{h ,w});

    IConvolutionLayer *fc1 = network->addConvolutionNd(*pool->getOutput(0), expandCh/4, DimsHW{1, 1},
                                                       weightMap[layerName + ".fc1.weight"], weightMap[layerName + ".fc1.bias"]);
    assert(fc1);
    IActivationLayer *relu = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu);
    IConvolutionLayer *fc2 = network->addConvolutionNd(*relu->getOutput(0), expandCh, DimsHW{1, 1},
                                                       weightMap[layerName + ".fc2.weight"], weightMap[layerName + ".fc2.bias"]);
    assert(fc2);
    IActivationLayer *hardSigmoid = network->addActivation(*fc2->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hardSigmoid);
    hardSigmoid->setAlpha(1.0 / 6.0);
    hardSigmoid->setBeta(0.5);

    IElementWiseLayer *ew = network->addElementWise(input, *hardSigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);

    return ew;
}

ILayer *seq1(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int outCh, int expandCh, int kSize,
             int stride, bool useSE, bool useHS, int w, string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, expandCh, DimsHW{1, 1}, weightMap[layerName + ".0.0.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setNbGroups(1);

    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".0.1", 1e-5);
    ILayer *act1;
    if (useHS) {
        // hard swish
        act1 = addHardWish(network, *bn1->getOutput(0));
    } else {
        // relu
        act1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(act1);
    }
    int p = (kSize - 1 ) / 2 * 1;
    IConvolutionLayer *conv2 = network->addConvolutionNd(*act1->getOutput(0), expandCh, DimsHW{kSize, kSize}, weightMap[layerName + ".1.0.weight"],emptyWt);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{p, p});
    conv2->setNbGroups(expandCh);
    IScaleLayer *bn2 = addBN(network, weightMap, *conv2->getOutput(0), layerName + ".1.1", 1e-5);
    ILayer *act2;
    if (useHS) {
        // hard swish
        act2 = addHardWish(network, *bn2->getOutput(0));
    } else {
        // relu
        act2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        assert(act2);
    }
    ITensor &tensor = *act2->getOutput(0);
    if (useSE) {
        ILayer *se = seLayer(network, weightMap, *act2->getOutput(0), expandCh, w, layerName + ".2");
        tensor = *se->getOutput(0);
    }
    IConvolutionLayer *conv3 = network->addConvolutionNd(tensor, outCh, DimsHW{1, 1}, weightMap[layerName + ".3.0.weight"], emptyWt);
    assert(conv3);
    IScaleLayer *bn3 = addBN(network, weightMap, *conv3->getOutput(0), layerName + ".3.1", 1e-5);

    return bn3;
}

ILayer *seq2(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int outCh, int expandCh, int kSize,
             int stride, bool useSE, bool useHS, int w, string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    int p = (kSize - 1) / 2;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, expandCh, DimsHW{kSize, kSize}, weightMap[layerName + ".0.0.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setNbGroups(expandCh);
    conv1->setPaddingNd(DimsHW{p, p});
    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".0.1", 1e-5);
    ILayer *act1;
    if (useHS) {
        act1 = addHardWish(network, *bn1->getOutput(0));
    } else {
        act1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    }
    ITensor &tensor = *act1->getOutput(0);
    if (useSE) {
        ILayer *se = seLayer(network, weightMap, *act1->getOutput(0), expandCh, w, layerName + ".1");
        tensor = *se->getOutput(0);
    }
    IConvolutionLayer *conv2 = network->addConvolutionNd(tensor, outCh, DimsHW{1, 1}, weightMap[layerName + ".2.0.weight"], emptyWt);
    assert(conv2);
    IScaleLayer *bn2 = addBN(network, weightMap, *conv2->getOutput(0), layerName + ".2.1", 1e-5);

    return bn2;
}

ILayer *invertedResidual(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int inCh, int outCh,
                         int kSize, int stride, int expandCh, bool useSE, bool useHS, int w, string layerName) {
    bool useResConnect = stride == 1 && inCh == outCh;
    ILayer *conv;
    if (expandCh != inCh) {
        conv = seq1(network, weightMap, input, outCh, expandCh, kSize, stride, useSE, useHS, w, layerName);
    } else {
        conv = seq2(network, weightMap, input, outCh, expandCh, kSize, stride, useSE, useHS, w, layerName);
    }
    if (useResConnect) {
        IElementWiseLayer *ew = network->addElementWise(*conv->getOutput(0), input, ElementWiseOperation::kSUM);
        return ew;
    }
    return conv;
}

ILayer *convBNActivation(INetworkDefinition *network, map<string, Weights> weightMap, ITensor &input, int outCh, int kSize, int stride, const string layerName, int g=1,  int dilation=1) {
    Weights empytWt{DataType::kFLOAT, nullptr, 0};
    int p = (kSize - 1) / 2 * dilation;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outCh, DimsHW{kSize, kSize}, weightMap[layerName + ".0.weight"], empytWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer *bn = addBN(network, weightMap, *conv1->getOutput(0), layerName+".1", 1e-5);

    ILayer *hardWish = addHardWish(network, *bn->getOutput(0));

    return hardWish;
}

// softmax layer
ILayer *reshapeSoftmax(INetworkDefinition *network, ITensor &input, int c) {
    IShuffleLayer *shuffleLayer1 = network->addShuffle(input);
    assert(shuffleLayer1);
    shuffleLayer1->setReshapeDimensions(Dims3{1, -1, c});

    Dims dim0 = shuffleLayer1->getOutput(0)->getDimensions();

    cout <<  "softmax output dims " << dim0.d[0] << " " << dim0.d[1] << " " << dim0.d[2] << " " << dim0.d[3] << endl;

    ISoftMaxLayer *softMaxLayer = network->addSoftMax(*shuffleLayer1->getOutput(0));
    assert(softMaxLayer);
    softMaxLayer->setAxes(1<<2);

    // 变为1维数组
    Dims dim_{};
    dim_.nbDims = 1;
    dim_.d[0] = -1;

    IShuffleLayer *shuffleLayer2 = network->addShuffle(*softMaxLayer->getOutput(0));
    assert(shuffleLayer2);
    shuffleLayer2->setReshapeDimensions(dim_);

    return shuffleLayer2;
}


ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config, DataType dataType, int maxBatchSize) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    ITensor *data = network->addInput(INPUT_BLOB_NAME, dataType, Dims3{maxBatchSize, INPUT_H, INPUT_W});
    assert(data);

    map<string, Weights> weightMap = loadWeight("driver_status_detection.wts");

    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    // construct network
    auto conv1 = convBNActivation(network, weightMap, *data, 16, 3, 2, "features.0");
    auto ir1 = invertedResidual(network, weightMap, *conv1->getOutput(0), 16, 16, 3,
                                2, 16, true, false, 56, "features.1.block");
    auto ir2 = invertedResidual(network, weightMap, *ir1->getOutput(0), 16, 24, 3,
                                2, 72, false, false, 28, "features.2.block");
    auto ir3 = invertedResidual(network, weightMap, *ir2->getOutput(0), 24, 24, 3,
                                1, 88, false, false, 28, "features.3.block");
    auto ir4 = invertedResidual(network, weightMap, *ir3->getOutput(0), 24, 40, 5,
                                2, 96, true, true, 14, "features.4.block");
    auto ir5 = invertedResidual(network, weightMap, *ir4->getOutput(0), 40, 40, 5,
                                1, 240, true, true, 14, "features.5.block");
    auto ir6 = invertedResidual(network, weightMap, *ir5->getOutput(0), 40, 40, 5,
                                1, 240, true, true, 14, "features.6.block");
    auto ir7 = invertedResidual(network, weightMap, *ir6->getOutput(0), 40, 48, 5,
                                1, 120, true, true, 14, "features.7.block");
    auto ir8 = invertedResidual(network, weightMap, *ir7->getOutput(0), 48, 48, 5,
                                1, 144, true, true, 14, "features.8.block");
    auto ir9 = invertedResidual(network, weightMap, *ir8->getOutput(0), 48, 96, 5,
                                2, 288, true, true, 7, "features.9.block");
    auto ir10 = invertedResidual(network, weightMap, *ir9->getOutput(0), 576, 96, 5,
                                 1, 96, true, true, 7, "features.10.block");
    auto ir11 = invertedResidual(network, weightMap, *ir10->getOutput(0), 96, 96, 5,
                                 1, 576, true, true, 7, "features.11.block");
    auto conv2 = convBNActivation(network, weightMap, *ir11->getOutput(0), 6 * 96, 1, 1, "features.12");
    auto pool = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7 ,7});
    assert(pool);
    pool->setStrideNd(DimsHW{7, 7});

    IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool->getOutput(0), 1024, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    auto hardSwish = addHardWish(network, *fc1->getOutput(0));
    // inference remove dropout
    IFullyConnectedLayer *fc2 = network->addFullyConnected(*hardSwish->getOutput(0), 10, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);

    ILayer *softMaxLayer = reshapeSoftmax(network, *fc2->getOutput(0), 10);

    softMaxLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    cout << "set output name" << endl;
    network->markOutput(*softMaxLayer->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 32);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    cout << "engine build" << endl;

    network->destroy();
    for (auto &mem : weightMap) {
        free((void *)(mem.second.values));
    }

    return engine;

}


void APIToModel(IHostMemory **modelStream, int maxBatchSize) {
    // create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    ICudaEngine *engine;

    engine = createEngine(builder, config, DataType::kFLOAT, maxBatchSize);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
    config->destroy();
}

int main(int argc, char** argv){
    if (argc !=2){
        cerr << "arguments are invalid" <<  endl;
    }

    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1])== "s") {
        IHostMemory *modelStream{nullptr};
        APIToModel(&modelStream, 1);
        assert(modelStream != nullptr);
        ofstream p("driver_status_detection.engine", ios::binary);
        if (!p) {
            cerr << "engine file error" << endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

        modelStream->destroy();
        return 0;
    }
}