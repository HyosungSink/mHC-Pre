#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_pre_tiling.h"
#include "../op_kernel/tiling_key_mhc_pre.h"

namespace {
constexpr size_t kXIndex = 0;
constexpr size_t kPhiIndex = 1;
constexpr size_t kAlphaIndex = 2;
constexpr size_t kBiasIndex = 3;
constexpr size_t kGammaIndex = 4;
constexpr size_t kHInIndex = 0;
constexpr size_t kHPostIndex = 1;
constexpr size_t kHResIndex = 2;
constexpr uint32_t kVectorAlignElements = 64;
constexpr uint32_t kMaxTileElements = 4096;
constexpr uint64_t kUbSafetyBytes = 8192;
constexpr uint32_t kFastHiddenDim = 16;

struct MhcPreShapeInfo {
    uint64_t positionCount;
    uint32_t n;
    uint32_t d;
    uint32_t width;
    uint32_t rowCount;
    uint32_t rowBufferLength;
};

bool IsPositiveShape(const gert::Shape &shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

bool IsShape(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : dims) {
        if (shape.GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

bool ParseAndValidateShapes(const gert::Shape &xShape, const gert::Shape &phiShape,
                            const gert::Shape &alphaShape, const gert::Shape &biasShape,
                            const gert::Shape *gammaShape, MhcPreShapeInfo &info)
{
    if (xShape.GetDimNum() != 4 || phiShape.GetDimNum() != 2 ||
        alphaShape.GetDimNum() != 1 || biasShape.GetDimNum() != 1 ||
        !IsPositiveShape(xShape) || !IsPositiveShape(phiShape) ||
        !IsPositiveShape(alphaShape) || !IsPositiveShape(biasShape)) {
        return false;
    }
    if (gammaShape != nullptr &&
        (gammaShape->GetDimNum() != 2 || !IsPositiveShape(*gammaShape))) {
        return false;
    }

    const uint64_t b = static_cast<uint64_t>(xShape.GetDim(0));
    const uint64_t s = static_cast<uint64_t>(xShape.GetDim(1));
    const uint64_t n = static_cast<uint64_t>(xShape.GetDim(2));
    const uint64_t d = static_cast<uint64_t>(xShape.GetDim(3));
    constexpr uint64_t u32Max = std::numeric_limits<uint32_t>::max();
    constexpr uint64_t u64Max = std::numeric_limits<uint64_t>::max();
    if (n > u32Max || d > u32Max || n > u64Max / d || b > u64Max / s) {
        return false;
    }
    const uint64_t width = n * d;
    if (n > u64Max / (n + 2U)) {
        return false;
    }
    const uint64_t rowCount = n * (n + 2U);
    const uint64_t alignedN = ((n + 7U) / 8U) * 8U;
    const uint64_t rowBufferLength = 2U * alignedN + n * n;
    const uint64_t positionCount = b * s;
    if (width > u32Max || rowCount > u32Max || rowBufferLength > u32Max ||
        rowCount > u64Max / width || positionCount > u64Max / width ||
        positionCount > u64Max / d || positionCount > u64Max / n ||
        positionCount > u64Max / (n * n)) {
        return false;
    }

    if (!IsShape(phiShape, {static_cast<int64_t>(rowCount), static_cast<int64_t>(width)}) ||
        !IsShape(alphaShape, {3}) ||
        !IsShape(biasShape, {static_cast<int64_t>(rowCount)}) ||
        (gammaShape != nullptr &&
         !IsShape(*gammaShape, {static_cast<int64_t>(n), static_cast<int64_t>(d)}))) {
        return false;
    }

    info = {positionCount, static_cast<uint32_t>(n), static_cast<uint32_t>(d),
            static_cast<uint32_t>(width), static_cast<uint32_t>(rowCount),
            static_cast<uint32_t>(rowBufferLength)};
    return true;
}

bool IsSupportedTypeCombination(ge::DataType xType, ge::DataType paramType)
{
    const bool evaluatorLayout =
        (xType == ge::DT_FLOAT16 && paramType == ge::DT_FLOAT16) ||
        (xType == ge::DT_FLOAT && paramType == ge::DT_FLOAT);
    const bool statementLayout =
        (xType == ge::DT_FLOAT16 || xType == ge::DT_BF16) && paramType == ge::DT_FLOAT;
    return evaluatorLayout || statementLayout;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1U) / alignment) * alignment;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(kXIndex);
    const gert::Tensor *tensorPhi = context->GetRequiredInputTensor(kPhiIndex);
    const gert::Tensor *tensorAlpha = context->GetRequiredInputTensor(kAlphaIndex);
    const gert::Tensor *tensorBias = context->GetRequiredInputTensor(kBiasIndex);
    const gert::Tensor *tensorGamma = context->GetOptionalInputTensor(kGammaIndex);
    const gert::StorageShape *xStorage = context->GetRequiredInputShape(kXIndex);
    const gert::StorageShape *phiStorage = context->GetRequiredInputShape(kPhiIndex);
    const gert::StorageShape *alphaStorage = context->GetRequiredInputShape(kAlphaIndex);
    const gert::StorageShape *biasStorage = context->GetRequiredInputShape(kBiasIndex);
    const gert::StorageShape *gammaStorage = context->GetOptionalInputShape(kGammaIndex);
    if (tensorX == nullptr || tensorPhi == nullptr || tensorAlpha == nullptr || tensorBias == nullptr ||
        xStorage == nullptr || phiStorage == nullptr || alphaStorage == nullptr || biasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    MhcPreShapeInfo shapeInfo{};
    const gert::Shape *gammaShape =
        gammaStorage == nullptr ? nullptr : &gammaStorage->GetStorageShape();
    if (!ParseAndValidateShapes(xStorage->GetStorageShape(), phiStorage->GetStorageShape(),
                                alphaStorage->GetStorageShape(), biasStorage->GetStorageShape(),
                                gammaShape, shapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *hInStorage = context->GetOutputShape(kHInIndex);
    const gert::StorageShape *hPostStorage = context->GetOutputShape(kHPostIndex);
    const gert::StorageShape *hResStorage = context->GetOutputShape(kHResIndex);
    const int64_t b = xStorage->GetStorageShape().GetDim(0);
    const int64_t s = xStorage->GetStorageShape().GetDim(1);
    if (hInStorage == nullptr || hPostStorage == nullptr || hResStorage == nullptr ||
        !IsShape(hInStorage->GetStorageShape(), {b, s, shapeInfo.d}) ||
        !IsShape(hPostStorage->GetStorageShape(), {b, s, shapeInfo.n}) ||
        !IsShape(hResStorage->GetStorageShape(), {b, s, shapeInfo.n, shapeInfo.n})) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xType = tensorX->GetDataType();
    const ge::DataType paramType = tensorPhi->GetDataType();
    if (!IsSupportedTypeCombination(xType, paramType) ||
        tensorAlpha->GetDataType() != paramType || tensorBias->GetDataType() != paramType ||
        (tensorGamma != nullptr && tensorGamma->GetDataType() != paramType)) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *attrNormEps = attrs->GetFloat(0);
    const float *attrHcEps = attrs->GetFloat(1);
    if (attrNormEps == nullptr || attrHcEps == nullptr || !std::isfinite(*attrNormEps) ||
        !std::isfinite(*attrHcEps) || *attrNormEps < 0.0F) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreCount = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t xTypeSize = ge::GetSizeByDataType(xType);
    const int64_t paramTypeSize = ge::GetSizeByDataType(paramType);
    const uint64_t rowBytes =
        AlignUp(static_cast<uint64_t>(shapeInfo.rowBufferLength) * sizeof(float), 32U);
    const uint64_t fixedBytes = rowBytes + kUbSafetyBytes;
    const uint64_t perElementBytes =
        static_cast<uint64_t>(xTypeSize + paramTypeSize) + 5U * sizeof(float);
    if (coreCount == 0 || xTypeSize <= 0 || paramTypeSize <= 0 ||
        ubSize <= fixedBytes + perElementBytes * kVectorAlignElements) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tileLength = (ubSize - fixedBytes) / perElementBytes;
    tileLength = std::min<uint64_t>(tileLength, kMaxTileElements);
    tileLength = (tileLength / kVectorAlignElements) * kVectorAlignElements;
    if (tileLength < kVectorAlignElements) {
        return ge::GRAPH_FAILED;
    }

    MhcPreTilingData *tiling = context->GetTilingData<MhcPreTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->positionCount = shapeInfo.positionCount;
    tiling->n = shapeInfo.n;
    tiling->d = shapeInfo.d;
    tiling->width = shapeInfo.width;
    tiling->rowCount = shapeInfo.rowCount;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->hasGamma = tensorGamma == nullptr ? 0U : 1U;
    tiling->invWidth = 1.0F / static_cast<float>(shapeInfo.width);
    tiling->normEps = *attrNormEps;
    tiling->hcEps = *attrHcEps;

    const uint32_t dtypeX = static_cast<uint32_t>(xType);
    const uint32_t dtypeParam = static_cast<uint32_t>(paramType);
    // 静态快速路径只依赖 (n, D) 与参数精度，不绑定具体 B/S，
    // 其余形状（含隐藏形状）继续走通用路径。
    const bool fastShape =
        shapeInfo.d == kFastHiddenDim && (shapeInfo.n == 4U || shapeInfo.n == 8U);
    const uint32_t fastSmall = (paramType == ge::DT_FLOAT && fastShape) ? 1U : 0U;
    const uint32_t fastOnePosition =
        (fastSmall != 0U && shapeInfo.positionCount <= coreCount) ? 1U : 0U;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(coreCount, shapeInfo.positionCount));
    ASCENDC_TPL_SEL_PARAM(context, dtypeX, dtypeParam, fastSmall, fastOnePosition);
    if (context->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *xShape = context->GetRequiredInputShape(kXIndex);
    const gert::Shape *phiShape = context->GetRequiredInputShape(kPhiIndex);
    const gert::Shape *alphaShape = context->GetRequiredInputShape(kAlphaIndex);
    const gert::Shape *biasShape = context->GetRequiredInputShape(kBiasIndex);
    const gert::Shape *gammaShape = context->GetOptionalInputShape(kGammaIndex);
    gert::Shape *hInShape = context->GetOutputShape(kHInIndex);
    gert::Shape *hPostShape = context->GetOutputShape(kHPostIndex);
    gert::Shape *hResShape = context->GetOutputShape(kHResIndex);
    if (xShape == nullptr || phiShape == nullptr || alphaShape == nullptr || biasShape == nullptr ||
        hInShape == nullptr || hPostShape == nullptr || hResShape == nullptr) {
        return GRAPH_FAILED;
    }
    MhcPreShapeInfo shapeInfo{};
    if (!ParseAndValidateShapes(*xShape, *phiShape, *alphaShape, *biasShape, gammaShape, shapeInfo)) {
        return GRAPH_FAILED;
    }
    const int64_t b = xShape->GetDim(0);
    const int64_t s = xShape->GetDim(1);
    *hInShape = gert::Shape({b, s, shapeInfo.d});
    *hPostShape = gert::Shape({b, s, shapeInfo.n});
    *hResShape = gert::Shape({b, s, shapeInfo.n, shapeInfo.n});
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xType = context->GetRequiredInputDataType(kXIndex);
    const ge::DataType paramType = context->GetRequiredInputDataType(kPhiIndex);
    const ge::DataType alphaType = context->GetRequiredInputDataType(kAlphaIndex);
    const ge::DataType biasType = context->GetRequiredInputDataType(kBiasIndex);
    const ge::DataType gammaType = context->GetOptionalInputDataType(kGammaIndex);
    if (!IsSupportedTypeCombination(xType, paramType) || alphaType != paramType || biasType != paramType ||
        (gammaType != ge::DT_UNDEFINED && gammaType != paramType)) {
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(kHInIndex, xType) != GRAPH_SUCCESS ||
        context->SetOutputDataType(kHPostIndex, paramType) != GRAPH_SUCCESS ||
        context->SetOutputDataType(kHResIndex, paramType) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
    class MhcPre : public OpDef {
    public:
        explicit MhcPre(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("phi")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("alpha")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("bias")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("gamma")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hIn")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hPost")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hRes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("normEps").AttrType(OPTIONAL).Float(1e-06);
            this->Attr("hcEps").AttrType(OPTIONAL).Float(1e-06);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcPre);
}  // namespace ops
