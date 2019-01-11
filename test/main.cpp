#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>

#include "BatchStream.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"

#include "anyconversion.h"
#include "config.h"
#include "AttributeTagTable.h"
#include "data_type.h"

#include "MemoryCounter.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::vector;

static Logger gLogger;

const std::string DATA_PATH = "../data/";

#define CalibrationMode 1 //Set to '0' for Legacy calibrator and any other value for Entropy calibrator

// Visualization
const float kVISUAL_THRESHOLD = 0.6f;

class Int8LegacyCalibrator : public nvinfer1::IInt8LegacyCalibrator
{
public:
    Int8LegacyCalibrator(BatchStream& stream, int firstBatch, double cutoff, double quantile, const char* networkName, bool readCache = true)
        : mStream(stream)
        , mFirstBatch(firstBatch)
        , mReadCache(readCache)
        , mNetworkName(networkName)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        reset(cutoff, quantile);
    }

    virtual ~Int8LegacyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }
    double getQuantile() const override { return mQuantile; }
    double getRegressionCutoff() const override { return mCutoff; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        std::cout << "get batch" << std::endl;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;

        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

    const void* readHistogramCache(size_t& length) override
    {
        length = mHistogramCache.size();
        return length ? &mHistogramCache[0] : nullptr;
    }

    void writeHistogramCache(const void* cache, size_t length) override
    {
        mHistogramCache.clear();
        std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
    }

    void reset(double cutoff, double quantile)
    {
        mCutoff = cutoff;
        mQuantile = quantile;
        mStream.reset(mFirstBatch);
    }

private:
    std::string calibrationTableName()
    {
        assert(mNetworkName != NULL);
        return std::string("CalibrationTable") + mNetworkName;
    }
    BatchStream mStream;
    int mFirstBatch;
    double mCutoff, mQuantile;
    bool mReadCache{true};
    const char* mNetworkName;
    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache, mHistogramCache;
};

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(const std::string& NetworkName, BatchStream& stream, int firstBatch, bool readCache = true)
        : NetworkName_(NetworkName)
        , mStream(stream)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], "data"));  // data is default
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);

        if(mReadCache)
            std::cout << "Read Calibration Cache" << std::endl;

        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::cout << "input file name : " << calibrationTableName() << std::endl;
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        std::cout << "output file name : " << calibrationTableName() << std::endl;
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    std::string calibrationTableName()
    {
        return std::string("CalibrationTable : ") + NetworkName_;
    }

    BatchStream mStream;
    size_t mInputCount;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
    std::string NetworkName_;
};

void caffeToTRTModel(const std::string& networkname,
                     const std::string& deployFile,           // Name for caffe prototxt
                     const std::string& modelFile,            // Name for model
                     const std::vector<std::string>& outputs, // Network outputs
                     unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                     unsigned int maxWorkSpace,
                     std::string mode,                        // Precision mode
                     bool read_cache,
                     int kCAL_BATCH_SIZE,
                     int kFIRST_CAL_BATCH,
                     int kNB_CAL_BATCHES,
                     IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    DataType dataType = DataType::kFLOAT;
    if (mode == "FP16")
        dataType = DataType::kHALF;
    std::cout << "Begin parsing model..." << std::endl;
    std::cout << mode << " mode running..." << std::endl;

    std::vector<std::string> data_directories{DATA_PATH + networkname + '/'};

    const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile, data_directories).c_str(),
                                                              locateFile(modelFile, data_directories).c_str(),
                                                              *network,
                                                              dataType);
    std::cout << "End parsing model..." << std::endl;

    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(maxWorkSpace);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    ICudaEngine* engine;
    if (mode == "INT8")
    {
#if CalibrationMode == 0
        std::cout << "Using Legacy Calibrator" << std::endl;
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
        calibrator.reset(new Int8LegacyCalibrator(calibrationStream, 0, kCUTOFF, kQUANTILE, gNetworkName, true));
#else
        std::cout << "Using Entropy Calibrator" << std::endl;
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", data_directories);
        calibrator.reset(new Int8EntropyCalibrator(networkname, calibrationStream, kFIRST_CAL_BATCH, read_cache));
#endif
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator.get());
    }
    else
    {
        builder->setFp16Mode(mode == "FP16");
    }
    std::cout << "Begin building engine..." << std::endl;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout << "End building engine..." << std::endl;

    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, const std::string& input_blob, const std::vector<std::string>& output_blob, float* inputData, int batchSize, std::vector<std::map<std::string, std::vector<float>>> &results)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    assert(engine.getNbBindings() == (1 + output_blob.size()));
    void* buffers[1 + output_blob.size()];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(input_blob.c_str());
    DimsCHW dims = static_cast<DimsCHW &&>(engine.getBindingDimensions(inputIndex));

    // Create GPU buffers and a stream
    CHECK_CUDA(cudaMalloc(&buffers[inputIndex], batchSize *  dims.c() * dims.h() * dims.w() * sizeof(float))); // Data

    for (int i = 0; i < output_blob.size(); i++) {
        int index = engine.getBindingIndex(output_blob[i].c_str());
        DimsCHW dims = static_cast<DimsCHW &&>(engine.getBindingDimensions(index));
        cudaMalloc(&buffers[index], batchSize * dims.c() * dims.h() * dims.w() * sizeof(float));
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * dims.c() * dims.h() * dims.w() * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    results.resize(batchSize);
    for(int i = 0; i < output_blob.size(); i++)
    {
        vector<float> output;
        string layername = output_blob[i];
        auto index = engine.getBindingIndex(layername.c_str());
        DimsCHW dims = static_cast<DimsCHW &&>(engine.getBindingDimensions(index));
        auto fea_len =  dims.c() * dims.h() * dims.w();
        output.resize(batchSize * fea_len);
        cudaMemcpyAsync(output.data(), buffers[index], batchSize * fea_len * sizeof(float), cudaMemcpyDeviceToHost, stream);

        for(int j = 0; j < batchSize; j++)
        {
            map<string, vector<float>> &feature = results[j];
            feature[layername].resize(fea_len);
            std::copy(output.begin() + j * fea_len, output.begin() + (j + 1) * fea_len, feature[layername].begin());
        }
    }

    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int i = 0; i < engine.getNbBindings(); ++i) {
        if (buffers[i] != nullptr) {
            CHECK_CUDA(cudaFree(buffers[i]));
        }
    }
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::vector<std::string> read_imagelist(std::string file_path) {
    if (hasEnding(file_path, ".ppm")) {
        LOG(WARNING) << "A image file is set as input.";
        return std::vector<std::string>{file_path};
    }
    std::ifstream input(file_path);
    if (!input.is_open()) {
        LOG(FATAL) << "Can not open the input imagelist.";
    }
    std::vector<std::string> imagenames;
    std::string tempname;
    while (input >> tempname) {
        if(tempname=="")
            continue;
        imagenames.push_back(tempname);
    }
    return imagenames;
}

void write_classify(std::ostream &out, std::string imagename, std::vector<BaseAttribute> &result,
                    int out_type = 0) {
    out << imagename;
    for (BaseAttribute att: result) {
        if (out_type == 0)
            out << " [" << att.idx << "]" << att.name << ":" << att.confidence;
        else if (out_type == 1)
            out << " " << att.name;
        else if (out_type == 2)
            out << " " << att.idx;
        else {
            LOG(FATAL) << "Not recognized out_type.";
        }
    }
    out << std::endl;
}


void printHelp()
{
    printf("Usage: ./TensorRT5-union --name MODELNAME\n");
    printf("Description: MODELDNAME is the name which you named in data/ directory\n");
    exit(0);
}

void parseOptions(int argc, char** argv)
{
    if(argc < 3)
        goto error;

    int i;
    for (i = 1; i < argc; i++)
    {
        char* optName = argv[i];
        if (0 == strcmp(optName, "--help"))
            goto error;
        else if (0 == strcmp(optName, "--name"))
        {
            if (++i == argc)
            {
                printf("Specify the name \n");
                goto error;
            }
        }
        else
            goto error;
    }
    return;

    error:
    printHelp();
}

int main(int argc, char** argv)
{
    parseOptions(argc, argv);

    Config model_config;
    std::string model_name = argv[2];

    model_config.Load("../config/" + model_name + ".json");
    std::string mode = model_config.getString("Mode");
    std::string prototxt_path = model_config.getString("PrototxtPath");
    std::string model_path = model_config.getString("ModelPath");
    std::string input_blob = model_config.getString("InputBlob");
    std::string network_name = model_config.getString("NetworkName");

    std::vector<std::string> output_blobs = model_config.getStringArray("OutputBlobs");
    std::vector<std::string> class_names = model_config.getStringArray("classNames");
    std::string category = model_config.getString("Category");

    int batch_size = model_config.getInteger("ModelBatchSize");
    int max_workspace = model_config.getInteger("MaxWorkspaceSize");
    int device_id = model_config.getInteger("DeviceNo");

    int input_height = model_config.getInteger("Height");
    int input_width = model_config.getInteger("Width");
    int input_channle = model_config.getInteger("Channel");
    std::vector<float> pixel_mean = model_config.getFloatArray("Mean");
    float pixel_scale = model_config.getFloat("Scale");

    bool read_cache = (bool) model_config.getInteger("readCalibrationCache");
    bool generate_model = (bool) model_config.getInteger("GenerateModel");
    bool profile = (bool) model_config.getInteger("Profile");
    bool verify = (bool) model_config.getInteger("Verify");

    int cal_batch_size = model_config.getInteger("CAL_BATCH_SIZE");
    int fisrt_cal_batch = model_config.getInteger("FIRST_CAL_BATCH");
    int nb_cal_batches = model_config.getInteger("NB_CAL_BATCHES");

    std::string test_list = model_config.getString("testImgFile");

    CHECK_CUDA(cudaSetDevice(device_id));

    // only for standard ssd
    if(category == "Detection")
        initLibNvInferPlugins(&gLogger, "");

    IHostMemory* trtModelStream{nullptr};
    std::ifstream cached_model;

    if(generate_model)
    {
        // Create a TensorRT model from the caffe model and serialize it to a stream
        caffeToTRTModel(network_name, prototxt_path, model_path, output_blobs, batch_size, max_workspace, mode, read_cache, cal_batch_size, fisrt_cal_batch, nb_cal_batches, &trtModelStream);

        std::ofstream model_file(DATA_PATH + network_name + '/' + network_name + '_' + mode + '_' + to_string(batch_size) + ".dat");
        model_file.write(reinterpret_cast<char *>(trtModelStream->data()), trtModelStream->size());
    }
    else
    {
        cached_model.open(DATA_PATH + network_name + '/' + network_name + '_' + mode + '_' + to_string(batch_size) + ".dat", std::ios::binary);
        if(!cached_model)
        {
            LOG(FATAL) << " falied to open file " << DATA_PATH + network_name + '/' + network_name + '_' + mode + '_' + to_string(batch_size) + ".dat";
        }
    }


    if(verify)
    {
        std::vector<samplesCommon::PPM> ppms(batch_size);

        vector<string> imageList = read_imagelist(locateFile(test_list, std::vector<std::string>{DATA_PATH + network_name + '/'}));

        assert(batch_size == imageList.size());

        for (int i = 0; i < batch_size; ++i)
        {
            ppms[i].init(input_channle, input_width, input_height);
            samplesCommon::readPPMFile(locateFile(imageList[i], std::vector<std::string>{DATA_PATH + network_name + '/'}), ppms[i]);
        }

        float* data = new float[batch_size * input_channle * input_height * input_width];

        for (int i = 0, volImg = input_channle * input_height * input_width; i < batch_size; ++i)
            for (int c = 0; c < input_channle; ++c)
                // The color image to input should be in BGR order
                for (unsigned j = 0, volChl = input_height * input_width; j < volChl; ++j)
                    data[i * volImg + c * volChl + j] = (float(ppms[i].buffer[j * input_channle + 2 - c]) - pixel_mean[c]) / pixel_scale;

        std::cout << "------deserializing------" << std::endl;
        MemoryCounter memoryCounter;

	    IRuntime* runtime = createInferRuntime(gLogger);
        // Deserialize the engine
        ICudaEngine* engine;
        if(generate_model)
        {
            engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
            trtModelStream->destroy();
        }
        else
        {
            cached_model.seekg (0, cached_model.end);
            int length = cached_model.tellg();
            cached_model.seekg (0, cached_model.beg);

            char * buffer = new char [length];
            cached_model.read(buffer,length);
            engine = runtime->deserializeCudaEngine(buffer, length, nullptr);
            delete[] buffer;
        }

        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();

	    memoryCounter.report("engine");

        SimpleProfiler profiler("engine");

        if(profile)
            context->setProfiler(&profiler);

        assert(context != nullptr);

        // Run inference
        std::vector<std::map<std::string, std::vector<float>>> results;

        doInference(*context, input_blob, output_blobs, data, batch_size, results);

        if(profile)
            std::cout << profiler;

        if(category == "Detection")
        {
            for (int p = 0; p < batch_size; ++p)
            {
                std::cout << " Image name:" << ppms[p].fileName.c_str();
                for(int i = 0; i < 100; ++i)
                {
                    float* det = results[p]["detection_out"].data() + 7 * i;
                    if (det[2] < kVISUAL_THRESHOLD)
                        continue;
                    assert((int) det[1] < class_names.size());
                    std::string storeName = class_names[(int) det[1]] + "-" + std::to_string(det[2]) + ".ppm";

                    std::cout << ", Label :" << class_names[(int) det[1]].c_str() << ","
                              << " confidence: " << det[2] * 100.f
                              << " xmin: " << det[3] * input_width
                              << " ymin: " << det[4] * input_height
                              << " xmax: " << det[5] * input_width
                              << " ymax: " << det[6] * input_height;
                }
                std::cout << std::endl;
            }
        }
        else if(category == "Classify")
        {
            AttributeTagTable attributeTagTable_;
            attributeTagTable_.loadFile("../config/" + model_name + ".cfg");
            vector<vector<BaseAttribute>> classify_results;

            for (size_t idx = 0; idx < results.size(); idx++) {
                std::vector<BaseAttribute> attrs;
                for (size_t a_idx = 0; a_idx < attributeTagTable_.tagtable_.size(); a_idx++) {
                    BaseAttribute attr;
                    attr.idx = attributeTagTable_.tagtable_[a_idx].index;
                    attr.name = attributeTagTable_.tagtable_[a_idx].tagname;
                    attr.thresh_low = attributeTagTable_.tagtable_[a_idx].threshold_lower;
                    attr.thresh_high = attributeTagTable_.tagtable_[a_idx].threshold_upper;
                    attr.categoryId = attributeTagTable_.tagtable_[a_idx].categoryId;
                    attr.mappingId = attributeTagTable_.tagtable_[a_idx].mappingId;
                    attr.confidence = results[idx][attributeTagTable_.tagtable_[a_idx].output_layer][attributeTagTable_.tagtable_[a_idx].output_id];

                    if (attr.confidence > attr.thresh_high) {
                        attrs.push_back(attr);
                    }
                }
                classify_results.push_back(attrs);
            }
            for (int index = 0; index < classify_results.size(); index++)
            {
                write_classify(std::cout, ppms[index].fileName, classify_results[index], 0);
            }
        }

        context->destroy();
        engine->destroy();
        runtime->destroy();

        delete[] data;
    }
    // Note: Once you call shutdownProtobufLibrary, you cannot use the parsers anymore.
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
