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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace nvinfer1;

static Logger gLogger;

static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 640;
static const int INPUT_H = 480;
static const int BATCH_SIZE = 1;
static const int OUTPUT_SIZE = 10;


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
        if (size == 432) {
            cout << name << endl;
        }
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
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));

    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    for (int i = 0; i < len; ++i) {
        shval[i] = bias[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    for (int i = 0; i < len; ++i) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;

    IScaleLayer *bn = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn);

    return bn;
}

ILayer *addHardSwish(INetworkDefinition *network, ITensor &input) {
    IActivationLayer *hardSigmoid = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hardSigmoid);
    hardSigmoid->setAlpha(1.0 / 6.0);
    hardSigmoid->setBeta(0.5);
    ILayer *ew = network->addElementWise(input, *hardSigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);

    return ew;
}

ILayer *seLayer(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int expandCh, int w, int h, string layerName) {
    IPoolingLayer *pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{h, w});
    assert(pool);
    pool->setStrideNd(DimsHW{h ,w});

    // ensure outCh % 8 == 0
    int outCh = max(expandCh / 4, 8);
    if (outCh % 8 != 0) {
        int mod = outCh % 8;
        outCh += (8 - mod);
    }
    assert(outCh % 8 == 0 && "seLayer outChannel wrong");

    IConvolutionLayer *fc1 = network->addConvolutionNd(*pool->getOutput(0), outCh, DimsHW{1, 1},
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
             int stride, bool useSE, bool useHS, int w, int h, string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, expandCh, DimsHW{1, 1}, weightMap[layerName + ".0.0.weight"], emptyWt);
    assert(conv1);
    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".0.1", 1e-5);
    ITensor *actTensor;
    if (useHS) {
        // hard swish
        ILayer *act1 = addHardSwish(network, *bn1->getOutput(0));
        actTensor = act1->getOutput(0);
    } else {
        // relu
        ILayer *act1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(act1);
        actTensor = act1->getOutput(0);
    }
    int p = (kSize - 1 ) / 2 * 1;
    IConvolutionLayer *conv2 = network->addConvolutionNd(*actTensor, expandCh, DimsHW{kSize, kSize}, weightMap[layerName + ".1.0.weight"],emptyWt);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{p, p});
    conv2->setNbGroups(expandCh);
    IScaleLayer *bn2 = addBN(network, weightMap, *conv2->getOutput(0), layerName + ".1.1", 1e-5);
    ITensor *hsTensor;
    hsTensor = nullptr;
    if (useHS) {
        // hard swish
        ILayer *act2 = addHardSwish(network, *bn2->getOutput(0));
        hsTensor = act2->getOutput(0);
    } else {
        // relu
        ILayer *act2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        assert(act2);
        hsTensor = act2->getOutput(0);
    }
    ITensor *tensor;
    tensor = nullptr;
    if (useSE) {
        ILayer *se = seLayer(network, weightMap, *hsTensor, expandCh, w, h, layerName + ".2");
        tensor = se->getOutput(0);
    } else {
        tensor = hsTensor;
    }
    string convName;
    string bnName;
    if (useSE) {
        convName = ".3.0.weight";
        bnName = ".3.1";
    } else {
        convName = ".2.0.weight";
        bnName = ".2.1";
    }
    IConvolutionLayer *conv3 = network->addConvolutionNd(*tensor, outCh, DimsHW{1, 1}, weightMap[layerName + convName], emptyWt);
    assert(conv3);
    IScaleLayer *bn3 = addBN(network, weightMap, *conv3->getOutput(0), layerName + bnName, 1e-5);

    return bn3;
}

ILayer *seq2(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int outCh, int expandCh, int kSize,
             int stride, bool useSE, bool useHS, int w, int h, string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    int p = (kSize - 1) / 2;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, expandCh, DimsHW{kSize, kSize}, weightMap[layerName + ".0.0.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setNbGroups(expandCh);
    conv1->setPaddingNd(DimsHW{p, p});
    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".0.1", 1e-5);
    ITensor *actTensor;
    if (useHS) {
        ILayer *act1 = addHardSwish(network, *bn1->getOutput(0));
        actTensor = act1->getOutput(0);
    } else {
        IActivationLayer *relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        actTensor = relu->getOutput(0);
    }
    ITensor *tensor;
    tensor = nullptr;
    if (useSE) {
        ILayer *se = seLayer(network, weightMap, *actTensor, expandCh, w, h,layerName + ".1");
        tensor = se->getOutput(0);
    } else {
        tensor = actTensor;
    }
    IConvolutionLayer *conv2 = network->addConvolutionNd(*tensor, outCh, DimsHW{1, 1}, weightMap[layerName + ".2.0.weight"], emptyWt);
    assert(conv2);
    IScaleLayer *bn2 = addBN(network, weightMap, *conv2->getOutput(0), layerName + ".2.1", 1e-5);

    return bn2;
}

ILayer *invertedResidual(INetworkDefinition *network, map<string, Weights>weightMap, ITensor &input, int inCh, int outCh,
                         int kSize, int stride, int expandCh, bool useSE, bool useHS, int w, int h, string layerName) {
    bool useResConnect = (stride == 1 && inCh == outCh);
    ILayer *conv;
    if (expandCh != inCh) {
        conv = seq1(network, weightMap, input, outCh, expandCh, kSize, stride, useSE, useHS, w, h,layerName);
    } else {
        conv = seq2(network, weightMap, input, outCh, expandCh, kSize, stride, useSE, useHS, w, h,layerName);
    }
    if (useResConnect) {
        IElementWiseLayer *ew = network->addElementWise( input, *conv->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return conv;
}

ILayer *convBNActivation(INetworkDefinition *network, map<string, Weights> weightMap, ITensor &input, int outCh, int kSize, int stride, const string layerName) {
    Weights empytWt{DataType::kFLOAT, nullptr, 0};
    int p = (kSize - 1) / 2 * 1;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outCh, DimsHW{kSize, kSize}, weightMap[layerName + ".0.weight"], empytWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(1);

    IScaleLayer *bn = addBN(network, weightMap, *conv1->getOutput(0), layerName+".1", 1e-5);

    ILayer *hardWish = addHardSwish(network, *bn->getOutput(0));

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

    ITensor *data = network->addInput(INPUT_BLOB_NAME, dataType, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    map<string, Weights> weightMap = loadWeight("driver_status_detection.wts");

    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    // construct network
    auto conv1 = convBNActivation(network, weightMap, *data, 16, 3, 2, "features.0");
    auto ir1 = invertedResidual(network, weightMap, *conv1->getOutput(0), 16, 16, 3,
                                2, 16, true, false, 160, 120, "features.1.block");
    auto ir2 = invertedResidual(network, weightMap, *ir1->getOutput(0), 16, 24, 3,
                                2, 72, false, false, 80, 60, "features.2.block");
    auto ir3 = invertedResidual(network, weightMap, *ir2->getOutput(0), 24, 24, 3,
                                1, 88, false, false, 80, 60, "features.3.block");
    auto ir4 = invertedResidual(network, weightMap, *ir3->getOutput(0), 24, 40, 5,
                                2, 96, true, true, 40, 30, "features.4.block");
    auto ir5 = invertedResidual(network, weightMap, *ir4->getOutput(0), 40, 40, 5,
                                1, 240, true, true, 40, 30, "features.5.block");
    auto ir6 = invertedResidual(network, weightMap, *ir5->getOutput(0), 40, 40, 5,
                                1, 240, true, true, 40, 30, "features.6.block");
    auto ir7 = invertedResidual(network, weightMap, *ir6->getOutput(0), 40, 48, 5,
                                1, 120, true, true, 40, 30, "features.7.block");
    auto ir8 = invertedResidual(network, weightMap, *ir7->getOutput(0), 48, 48, 5,
                                1, 144, true, true, 40, 30, "features.8.block");
    auto ir9 = invertedResidual(network, weightMap, *ir8->getOutput(0), 48, 96, 5,
                                2, 288, true, true, 20, 15, "features.9.block");
    auto ir10 = invertedResidual(network, weightMap, *ir9->getOutput(0), 96, 96, 5,
                                 1, 576, true, true, 20, 15, "features.10.block");
    auto ir11 = invertedResidual(network, weightMap, *ir10->getOutput(0), 96, 96, 5,
                                 1, 576, true, true, 20, 15, "features.11.block");
    auto conv2 = convBNActivation(network, weightMap, *ir11->getOutput(0), 576, 1, 1, "features.12");
    auto pool = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kAVERAGE, DimsHW{15 ,20});
    assert(pool);
    pool->setStrideNd(DimsHW{15, 20});

    IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool->getOutput(0), 1024, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    auto hardSwish = addHardSwish(network, *fc1->getOutput(0));
    // inference remove dropout
    IFullyConnectedLayer *fc2 = network->addFullyConnected(*hardSwish->getOutput(0), 10, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);

    ILayer *softMaxLayer = reshapeSoftmax(network, *fc2->getOutput(0), 10);
    softMaxLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    cout << "set output name" << endl;
    network->markOutput(*softMaxLayer->getOutput(0));

//    fc2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    cout << "set output name" << endl;
//    network->markOutput(*fc2->getOutput(0));



    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
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

void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    //create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //release stream and bufffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv){
    if (argc !=2){
        cerr << "arguments are invalid" <<  endl;
    }

    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1])== "-s") {
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
    } else {
        ifstream file("driver_status_detection.engine", ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char [size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    // input data
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    cv::Mat im = cv::imread("img_1.jpg");
    if (im.empty()) {
        cerr << "image file error" << endl;
    }
    cv::normalize(im, im, 1.0, 0.0, cv::NORM_MINMAX);
//    float *p_data = &data[0];
//    for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
//        p_data[i] = im.at<cv::Vec3b>(i)[0] / 255.0;
//        p_data[i + INPUT_H * INPUT_W] = im.at<cv::Vec3b>(i)[1] / 255.0;
//        p_data[i + 2 * INPUT_H * INPUT_W] = im.at<cv::Vec3b>(i)[2] / 255.0;
//    }
    unsigned vol = INPUT_H * INPUT_W * 3;
    auto fileDataChar = (uchar *)malloc(BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(uchar));
    fileDataChar = im.data;
    for (int i = 0; i < vol; ++i) {
        data[i] = (float)fileDataChar[i] * 1.0;
    }

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine);

    IExecutionContext *context = engine->createExecutionContext();
    assert(context);

    delete[] trtModelStream;

    // inference
    float prob[BATCH_SIZE * OUTPUT_SIZE];
    auto start = chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = chrono::system_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    //free engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // output
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cout << prob[i] << " ";
    }
    cout << endl;
}