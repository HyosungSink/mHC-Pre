#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_mhc_pre.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {
constexpr int32_t kDeviceId = 0;

struct CaseConfig {
    std::string caseId;
    int64_t b = 0;
    int64_t s = 0;
    int64_t n = 0;
    int64_t d = 0;
    aclDataType xType = ACL_DT_UNDEFINED;
    aclDataType paramType = ACL_DT_UNDEFINED;
    bool hasGamma = false;
    double normEps = 0.0;
    double hcEps = 0.0;
    float absTolerance = 0.0F;
    float relTolerance = 0.0F;
};

struct DeviceBuffer {
    void *address = nullptr;
    size_t size = 0;
};

struct CaseResult {
    std::string caseId;
    std::string status = "RE";
    double localUs = 0.0;
    bool hasTiming = false;
    float maxAbsError = 0.0F;
    size_t comparedElements = 0;
    size_t mismatchedElements = 0;
    std::vector<float> samplesUs;
};

struct TensorSet {
    DeviceBuffer x;
    DeviceBuffer phi;
    DeviceBuffer alpha;
    DeviceBuffer bias;
    DeviceBuffer gamma;
    DeviceBuffer hIn;
    DeviceBuffer hPost;
    DeviceBuffer hRes;
    aclTensor *xTensor = nullptr;
    aclTensor *phiTensor = nullptr;
    aclTensor *alphaTensor = nullptr;
    aclTensor *biasTensor = nullptr;
    aclTensor *gammaTensor = nullptr;
    aclTensor *hInTensor = nullptr;
    aclTensor *hPostTensor = nullptr;
    aclTensor *hResTensor = nullptr;
};

std::string Join(const std::string &root, const std::string &name)
{
    return root + "/" + name;
}

aclDataType ParseType(const std::string &value)
{
    if (value == "float16") {
        return ACL_FLOAT16;
    }
    if (value == "bfloat16") {
        return ACL_BF16;
    }
    if (value == "float32") {
        return ACL_FLOAT;
    }
    return ACL_DT_UNDEFINED;
}

size_t TypeSize(aclDataType type)
{
    return type == ACL_FLOAT ? sizeof(float) : sizeof(uint16_t);
}

bool LoadConfig(const std::string &directory, CaseConfig &config)
{
    std::ifstream input(Join(directory, "case.meta"));
    std::string key;
    while (input >> key) {
        if (key == "case_id") {
            input >> config.caseId;
        } else if (key == "b") {
            input >> config.b;
        } else if (key == "s") {
            input >> config.s;
        } else if (key == "n") {
            input >> config.n;
        } else if (key == "d") {
            input >> config.d;
        } else if (key == "x_dtype") {
            std::string value;
            input >> value;
            config.xType = ParseType(value);
        } else if (key == "param_dtype") {
            std::string value;
            input >> value;
            config.paramType = ParseType(value);
        } else if (key == "has_gamma") {
            int value = 0;
            input >> value;
            config.hasGamma = value != 0;
        } else if (key == "norm_eps") {
            input >> config.normEps;
        } else if (key == "hc_eps") {
            input >> config.hcEps;
        } else if (key == "abs_tolerance") {
            input >> config.absTolerance;
        } else if (key == "rel_tolerance") {
            input >> config.relTolerance;
        } else {
            return false;
        }
    }
    return input.eof() && !config.caseId.empty() && config.b > 0 && config.s > 0 &&
           config.n > 0 && config.d > 0 && config.xType != ACL_DT_UNDEFINED &&
           config.paramType != ACL_DT_UNDEFINED;
}

bool LoadBinary(const std::string &path, std::vector<uint8_t> &data)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    return size == 0 || static_cast<bool>(
        input.read(reinterpret_cast<char *>(data.data()), size));
}

float DecodeValue(const std::vector<uint8_t> &data, aclDataType type, size_t index)
{
    if (type == ACL_FLOAT) {
        float value = 0.0F;
        std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits = 0;
    std::memcpy(&bits, data.data() + index * sizeof(bits), sizeof(bits));
    if (type == ACL_BF16) {
        const uint32_t fp32Bits = static_cast<uint32_t>(bits) << 16U;
        float value = 0.0F;
        std::memcpy(&value, &fp32Bits, sizeof(value));
        return value;
    }
    __fp16 value{};
    std::memcpy(&value, &bits, sizeof(bits));
    return static_cast<float>(value);
}

bool AllocateInput(DeviceBuffer &buffer, const std::vector<uint8_t> &host)
{
    buffer.size = host.size();
    return buffer.size > 0 &&
           aclrtMalloc(&buffer.address, buffer.size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
           aclrtMemcpy(buffer.address, buffer.size, host.data(), host.size(),
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool AllocateOutput(DeviceBuffer &buffer, size_t size)
{
    buffer.size = size;
    return size > 0 &&
           aclrtMalloc(&buffer.address, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType type, void *address)
{
    return aclCreateTensor(shape.data(), shape.size(), type, nullptr, 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), address);
}

void ReleaseBuffer(DeviceBuffer &buffer)
{
    if (buffer.address != nullptr) {
        aclrtFree(buffer.address);
        buffer.address = nullptr;
    }
}

void ReleaseTensors(TensorSet &tensors)
{
    if (tensors.xTensor != nullptr) aclDestroyTensor(tensors.xTensor);
    if (tensors.phiTensor != nullptr) aclDestroyTensor(tensors.phiTensor);
    if (tensors.alphaTensor != nullptr) aclDestroyTensor(tensors.alphaTensor);
    if (tensors.biasTensor != nullptr) aclDestroyTensor(tensors.biasTensor);
    if (tensors.gammaTensor != nullptr) aclDestroyTensor(tensors.gammaTensor);
    if (tensors.hInTensor != nullptr) aclDestroyTensor(tensors.hInTensor);
    if (tensors.hPostTensor != nullptr) aclDestroyTensor(tensors.hPostTensor);
    if (tensors.hResTensor != nullptr) aclDestroyTensor(tensors.hResTensor);
    ReleaseBuffer(tensors.x);
    ReleaseBuffer(tensors.phi);
    ReleaseBuffer(tensors.alpha);
    ReleaseBuffer(tensors.bias);
    ReleaseBuffer(tensors.gamma);
    ReleaseBuffer(tensors.hIn);
    ReleaseBuffer(tensors.hPost);
    ReleaseBuffer(tensors.hRes);
}

bool LoadAndAllocate(const std::string &directory, const CaseConfig &config, TensorSet &tensors)
{
    std::vector<uint8_t> x;
    std::vector<uint8_t> phi;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> bias;
    std::vector<uint8_t> gamma;
    if (!LoadBinary(Join(directory, "x.bin"), x) ||
        !LoadBinary(Join(directory, "phi.bin"), phi) ||
        !LoadBinary(Join(directory, "alpha.bin"), alpha) ||
        !LoadBinary(Join(directory, "bias.bin"), bias) ||
        (config.hasGamma && !LoadBinary(Join(directory, "gamma.bin"), gamma))) {
        return false;
    }
    const int64_t positions = config.b * config.s;
    const int64_t width = config.n * config.d;
    const int64_t rows = config.n * config.n + 2 * config.n;
    if (x.size() != static_cast<size_t>(positions * width) * TypeSize(config.xType) ||
        phi.size() != static_cast<size_t>(rows * width) * TypeSize(config.paramType) ||
        alpha.size() != 3U * TypeSize(config.paramType) ||
        bias.size() != static_cast<size_t>(rows) * TypeSize(config.paramType) ||
        (config.hasGamma &&
         gamma.size() != static_cast<size_t>(width) * TypeSize(config.paramType))) {
        return false;
    }
    if (!AllocateInput(tensors.x, x) || !AllocateInput(tensors.phi, phi) ||
        !AllocateInput(tensors.alpha, alpha) || !AllocateInput(tensors.bias, bias) ||
        (config.hasGamma && !AllocateInput(tensors.gamma, gamma)) ||
        !AllocateOutput(tensors.hIn,
                        static_cast<size_t>(positions * config.d) * TypeSize(config.xType)) ||
        !AllocateOutput(tensors.hPost,
                        static_cast<size_t>(positions * config.n) * TypeSize(config.paramType)) ||
        !AllocateOutput(tensors.hRes,
                        static_cast<size_t>(positions * config.n * config.n) *
                            TypeSize(config.paramType))) {
        return false;
    }

    const std::vector<int64_t> xShape{config.b, config.s, config.n, config.d};
    const std::vector<int64_t> phiShape{rows, width};
    const std::vector<int64_t> alphaShape{3};
    const std::vector<int64_t> biasShape{rows};
    const std::vector<int64_t> gammaShape{config.n, config.d};
    const std::vector<int64_t> hInShape{config.b, config.s, config.d};
    const std::vector<int64_t> hPostShape{config.b, config.s, config.n};
    const std::vector<int64_t> hResShape{config.b, config.s, config.n, config.n};
    tensors.xTensor = CreateTensor(xShape, config.xType, tensors.x.address);
    tensors.phiTensor = CreateTensor(phiShape, config.paramType, tensors.phi.address);
    tensors.alphaTensor = CreateTensor(alphaShape, config.paramType, tensors.alpha.address);
    tensors.biasTensor = CreateTensor(biasShape, config.paramType, tensors.bias.address);
    if (config.hasGamma) {
        tensors.gammaTensor = CreateTensor(gammaShape, config.paramType, tensors.gamma.address);
    }
    tensors.hInTensor = CreateTensor(hInShape, config.xType, tensors.hIn.address);
    tensors.hPostTensor = CreateTensor(hPostShape, config.paramType, tensors.hPost.address);
    tensors.hResTensor = CreateTensor(hResShape, config.paramType, tensors.hRes.address);
    return tensors.xTensor != nullptr && tensors.phiTensor != nullptr &&
           tensors.alphaTensor != nullptr && tensors.biasTensor != nullptr &&
           (!config.hasGamma || tensors.gammaTensor != nullptr) &&
           tensors.hInTensor != nullptr && tensors.hPostTensor != nullptr &&
           tensors.hResTensor != nullptr;
}

bool PrepareInvocation(const CaseConfig &config, const TensorSet &tensors,
                       void *&workspace, uint64_t &workspaceSize, aclOpExecutor *&executor)
{
    workspace = nullptr;
    workspaceSize = 0;
    executor = nullptr;
    if (aclnnMhcPreGetWorkspaceSize(
            tensors.xTensor, tensors.phiTensor, tensors.alphaTensor, tensors.biasTensor,
            tensors.gammaTensor, config.normEps, config.hcEps, tensors.hInTensor,
            tensors.hPostTensor, tensors.hResTensor, &workspaceSize, &executor) != 0) {
        return false;
    }
    return workspaceSize == 0 ||
           aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

void ReleaseInvocation(void *workspace, aclOpExecutor *executor)
{
    if (workspace != nullptr) aclrtFree(workspace);
    if (executor != nullptr) aclDestroyAclOpExecutor(executor);
}

bool LaunchOnce(const CaseConfig &config, const TensorSet &tensors, aclrtStream stream)
{
    void *workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool ok = PrepareInvocation(config, tensors, workspace, workspaceSize, executor) &&
              aclnnMhcPre(workspace, workspaceSize, executor, stream) == 0 &&
              aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    ReleaseInvocation(workspace, executor);
    return ok;
}

bool CompareOutput(const DeviceBuffer &device, const std::string &goldenPath,
                   aclDataType type, const CaseConfig &config, CaseResult &result)
{
    std::vector<uint8_t> actual(device.size);
    std::vector<uint8_t> golden;
    if (!LoadBinary(goldenPath, golden) || golden.size() != actual.size() ||
        aclrtMemcpy(actual.data(), actual.size(), device.address, device.size,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    const size_t count = actual.size() / TypeSize(type);
    result.comparedElements += count;
    for (size_t index = 0; index < count; ++index) {
        const float observed = DecodeValue(actual, type, index);
        const float expected = DecodeValue(golden, type, index);
        const float error = std::fabs(observed - expected);
        result.maxAbsError = std::max(result.maxAbsError, error);
        if (!std::isfinite(observed) ||
            error > config.absTolerance + config.relTolerance * std::fabs(expected)) {
            ++result.mismatchedElements;
        }
    }
    return true;
}

bool Measure(const CaseConfig &config, const TensorSet &tensors, aclrtStream stream,
             int warmupRuns, int timedRuns, std::vector<float> &samples)
{
    for (int index = 0; index < warmupRuns; ++index) {
        if (!LaunchOnce(config, tensors, stream)) {
            return false;
        }
    }
    bool ok = true;
    for (int index = 0; index < timedRuns && ok; ++index) {
        void *workspace = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ok = PrepareInvocation(config, tensors, workspace, workspaceSize, executor) &&
             aclrtCreateEvent(&start) == ACL_SUCCESS &&
             aclrtCreateEvent(&end) == ACL_SUCCESS &&
             aclrtRecordEvent(start, stream) == ACL_SUCCESS &&
             aclnnMhcPre(workspace, workspaceSize, executor, stream) == 0 &&
             aclrtRecordEvent(end, stream) == ACL_SUCCESS &&
             aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        if (ok) {
            float elapsedMs = 0.0F;
            ok = aclrtEventElapsedTime(&elapsedMs, start, end) == ACL_SUCCESS;
            if (ok) samples.push_back(elapsedMs * 1000.0F);
        }
        if (start != nullptr) aclrtDestroyEvent(start);
        if (end != nullptr) aclrtDestroyEvent(end);
        ReleaseInvocation(workspace, executor);
    }
    return ok && samples.size() == static_cast<size_t>(timedRuns);
}

CaseResult RunCase(const std::string &directory, aclrtStream stream,
                   int warmupRuns, int timedRuns)
{
    CaseResult result;
    CaseConfig config;
    TensorSet tensors;
    if (!LoadConfig(directory, config)) {
        result.caseId = directory;
        return result;
    }
    result.caseId = config.caseId;
    if (!LoadAndAllocate(directory, config, tensors)) {
        ReleaseTensors(tensors);
        return result;
    }
    if (!LaunchOnce(config, tensors, stream)) {
        ReleaseTensors(tensors);
        return result;
    }
    const bool inRead = CompareOutput(tensors.hIn, Join(directory, "h_in_golden.bin"),
                                      config.xType, config, result);
    const bool postRead = CompareOutput(tensors.hPost, Join(directory, "h_post_golden.bin"),
                                        config.paramType, config, result);
    const bool resRead = CompareOutput(tensors.hRes, Join(directory, "h_res_golden.bin"),
                                       config.paramType, config, result);
    if (!inRead || !postRead || !resRead || result.mismatchedElements != 0) {
        result.status = !inRead || !postRead || !resRead ? "RE" : "WA";
        ReleaseTensors(tensors);
        return result;
    }
    if (!Measure(config, tensors, stream, warmupRuns, timedRuns, result.samplesUs)) {
        ReleaseTensors(tensors);
        return result;
    }
    std::sort(result.samplesUs.begin(), result.samplesUs.end());
    const size_t middle = result.samplesUs.size() / 2U;
    result.localUs = result.samplesUs.size() % 2U == 0U
        ? (result.samplesUs[middle - 1U] + result.samplesUs[middle]) / 2.0
        : result.samplesUs[middle];
    result.hasTiming = true;
    result.status = "Pass";
    ReleaseTensors(tensors);
    return result;
}

bool SaveResults(const std::string &path, const std::vector<CaseResult> &results,
                 int warmupRuns, int timedRuns, bool runtimeReady)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        return false;
    }
    std::string overallStatus = runtimeReady ? "Pass" : "RE";
    for (const auto &result : results) {
        if (result.status == "RE") {
            overallStatus = "RE";
        } else if (overallStatus == "Pass" && result.status != "Pass") {
            overallStatus = result.status;
        }
    }
    output << std::setprecision(10);
    output << "{\n  \"schema\": \"mhc_pre_runner_result/v1\",\n";
    output << "  \"status\": \"" << overallStatus << "\",\n";
    output << "  \"measurement\": \"aclrtEventElapsedTime_device_us\",\n";
    output << "  \"warmup_runs\": " << warmupRuns << ",\n";
    output << "  \"timed_runs\": " << timedRuns << ",\n";
    output << "  \"cases\": [\n";
    for (size_t index = 0; index < results.size(); ++index) {
        const auto &result = results[index];
        output << "    {\"case_id\": \"" << result.caseId << "\", \"status\": \""
               << result.status << "\", \"local_us\": ";
        if (result.hasTiming) output << result.localUs;
        else output << "null";
        output << ", \"max_abs_error\": " << result.maxAbsError
               << ", \"compared_elements\": " << result.comparedElements
               << ", \"mismatched_elements\": " << result.mismatchedElements
               << ", \"local_accuracy_error_percent\": ";
        if (result.comparedElements != 0U) {
            output << 100.0 * static_cast<double>(result.mismatchedElements) /
                          static_cast<double>(result.comparedElements);
        } else {
            output << "null";
        }
        output << ", \"samples_us\": [";
        for (size_t sample = 0; sample < result.samplesUs.size(); ++sample) {
            if (sample != 0U) output << ", ";
            output << result.samplesUs[sample];
        }
        output << "]}" << (index + 1U == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return static_cast<bool>(output);
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 5) {
        return 2;
    }
    const std::string resultPath = argv[1];
    const int warmupRuns = std::stoi(argv[2]);
    const int timedRuns = std::stoi(argv[3]);
    std::vector<CaseResult> results;
    bool runtimeReady = warmupRuns >= 0 && timedRuns > 0 &&
                        aclnnInit(nullptr) == 0 &&
                        aclrtSetDevice(kDeviceId) == ACL_SUCCESS;
    aclrtStream stream = nullptr;
    runtimeReady = runtimeReady && aclrtCreateStream(&stream) == ACL_SUCCESS;
    if (runtimeReady) {
        for (int index = 4; index < argc; ++index) {
            results.push_back(RunCase(argv[index], stream, warmupRuns, timedRuns));
        }
    }
    if (stream != nullptr) aclrtDestroyStream(stream);
    if (runtimeReady) aclrtResetDevice(kDeviceId);
    aclnnFinalize();
    const bool saved = SaveResults(resultPath, results, warmupRuns, timedRuns, runtimeReady);
    bool passed = saved && runtimeReady && !results.empty();
    for (const auto &result : results) passed = passed && result.status == "Pass";
    return passed ? 0 : 3;
}
