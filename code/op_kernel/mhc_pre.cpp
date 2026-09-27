#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

// 静态形状快速路径：D=16 且 n 为 4/8 时展开为编译期常量，
// 行缓冲按 32 字节分组排布，使 pPre/pPost/pRes 三段全部对齐，
// 从而避免 Gather 与标量-向量往返同步。
template <class DT_X, class DT_PARAM, uint32_t STATIC_N, bool ONE_POSITION,
          bool EPS_DEFAULT>
class KernelMhcPreSmallStatic {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                GM_ADDR tiling)
    {
        // n/D/invWidth 由模板定下，gamma 由可选输入指针判定，eps 常见取值走 TilingKey：
        // 单位置 + 默认 eps 时本路径完全不访问 Tiling 缓冲。
        // Init 只做地址与指针准备，任何冷 GM 取数都延后到搬运下发之后。
        hasGamma_ = gamma != nullptr;
        tilingGm_ = tiling;
        if constexpr (EPS_DEFAULT) {
            normEps_ = kDefaultEps;
            hcEps_ = kDefaultEps;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(bias));
        if (hasGamma_) {
            gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(gamma));
        }
        hInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(hIn));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hPost));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hRes));

        if constexpr (!IsSameType<DT_X, float>::value) {
            xRaw_ = LocalTensor<DT_X>(TPosition::VECIN, kXRawAddress, kWidth);
            xOutRaw_ = LocalTensor<DT_X>(TPosition::VECOUT, kXOutAddress, kD);
        }
        xFloat_ = LocalTensor<float>(TPosition::VECIN, kXFloatAddress, kWidth);
        phiFloat_ = LocalTensor<float>(TPosition::VECIN, kPhiFloatAddress, kPhiLength);
        biasFloat_ = LocalTensor<float>(TPosition::VECIN, kBiasFloatAddress, kRowLength);
        gammaFloat_ = LocalTensor<float>(TPosition::VECIN, kGammaFloatAddress, kWidth);
        product_ = LocalTensor<float>(TPosition::VECCALC, kProductAddress, kProductLength);
        hInAcc_ = LocalTensor<float>(TPosition::VECOUT, kHInAddress, kWidth);
        reduceOut_ = LocalTensor<float>(TPosition::VECCALC, kReduceAddress, 8U);
        sigmoidTmp_ = LocalTensor<float>(TPosition::VECCALC, kSigmoidAddress, 16U);
        postScale_ = LocalTensor<float>(TPosition::VECCALC, kPostScaleAddress, 8U);
        postEps_ = LocalTensor<float>(TPosition::VECCALC, kPostEpsAddress, 8U);
        rowAcc_ = LocalTensor<float>(TPosition::VECOUT, kRowAddress, kRowLength);
    }

    __aicore__ inline void Process()
    {
        const uint64_t blockIndex = GetBlockIdx();
        // x 的搬运先下发，再下发常量搬运，两笔冷访存的时延由同一次
        // MTE2->V 等待覆盖，而不是一前一后串起来。
        IssueXLoad(blockIndex);
        LoadConstants();
        if constexpr (ONE_POSITION) {
            ProcessLoaded(blockIndex);
        } else {
            const uint64_t blockCount = GetBlockNum();
            for (uint64_t position = blockIndex; position < positionCount_;
                 position += blockCount) {
                if (position != blockIndex) {
                    SyncMte3ToVector();
                    SyncVectorToMte2();
                    IssueXLoad(position);
                    SyncMte2ToVector();
                }
                ProcessLoaded(position);
            }
        }
        ResetMask();
    }

private:
    // hPost 经由 hIn 的 Brcb 输出取出，Brcb 单次只展开 8 个 lane，
    // 因此该结构要求 2n <= 8；n=8 使用 KernelMhcPreLegacyN8。
    static_assert(STATIC_N == 4U);
    static constexpr uint32_t kN = STATIC_N;
    static constexpr uint32_t kD = 16U;
    static constexpr uint32_t kWidth = kN * kD;
    static constexpr uint32_t kRows = kN * (kN + 2U);
    static constexpr uint32_t kPhiLength = kWidth * kRows;
    static constexpr uint32_t kWidthRepeats = kWidth / 64U;
    static constexpr uint32_t kProductLength = 64U * kRows;
    static constexpr float kInvWidth = 1.0F / static_cast<float>(kWidth);
    static constexpr float kDefaultEps = 1e-6F;
    // 行缓冲连续排布（与 GM 中 bias 的顺序一致）：pPre@0、pPost@kN、pRes@2kN。
    // pRes 起点 2n 个 float 是 32 字节整数倍；pPost 不必对齐，它只经由
    // hIn 本来就要做的 Brcb 取出，再用跨步搬运写回。
    static constexpr uint32_t kPostLocal = kN;
    static constexpr uint32_t kResLocal = 2U * kN;
    static constexpr uint32_t kRowLength = kRows;
    static constexpr uint32_t kRowBlocks = kRowLength / 8U;
    static constexpr bool kRawX = !IsSameType<DT_X, float>::value;

    static constexpr uint32_t Align32(uint32_t value)
    {
        return (value + 31U) & ~31U;
    }

    static constexpr uint32_t kXRawAddress = 0U;
    static constexpr uint32_t kXFloatAddress =
        Align32(kXRawAddress + (kRawX ? kWidth * sizeof(DT_X) : 0U));
    static constexpr uint32_t kPhiFloatAddress = Align32(kXFloatAddress + kWidth * 4U);
    static constexpr uint32_t kBiasFloatAddress =
        Align32(kPhiFloatAddress + kPhiLength * 4U);
    static constexpr uint32_t kGammaFloatAddress =
        Align32(kBiasFloatAddress + kRowLength * 4U);
    static constexpr uint32_t kProductAddress = Align32(kGammaFloatAddress + kWidth * 4U);
    static constexpr uint32_t kHInAddress = Align32(kProductAddress + kProductLength * 4U);
    static constexpr uint32_t kReduceAddress = Align32(kHInAddress + kWidth * 4U);
    static constexpr uint32_t kSigmoidAddress = kReduceAddress + 32U;
    static constexpr uint32_t kPostScaleAddress = Align32(kSigmoidAddress + 16U * 4U);
    static constexpr uint32_t kPostEpsAddress = kPostScaleAddress + 32U;
    static constexpr uint32_t kRowAddress = kPostEpsAddress + 32U;
    static constexpr uint32_t kXOutAddress = Align32(kRowAddress + kRowLength * 4U);

    __aicore__ inline void SyncMte2ToVector()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void SyncMte3ToVector()
    {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void LoadConstants()
    {
        const DataCopyExtParams phiCopy{1, kPhiLength * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams biasCopy{1, kRows * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams gammaCopy{1, kWidth * sizeof(DT_PARAM), 0, 0, 0};
        DataCopyPad(phiFloat_, phiGm_, phiCopy, {});
        DataCopyPad(biasFloat_, biasGm_, biasCopy, {});
        if (hasGamma_) {
            DataCopyPad(gammaFloat_, gammaGm_, gammaCopy, {});
        }
        // 搬运已经下发，现在才做冷标量取数：alpha 的读取时延被上面的 MTE2 掩盖。
        // 这些值直到每个位置的 Transform 阶段才会被消费。
        alpha0_ = static_cast<float>(alphaGm_.GetValue(0U));
        alpha1_ = static_cast<float>(alphaGm_.GetValue(1U));
        alpha2_ = static_cast<float>(alphaGm_.GetValue(2U));
        if constexpr (!EPS_DEFAULT || !ONE_POSITION) {
            __gm__ MhcPreTilingData *tilingPtr =
                reinterpret_cast<__gm__ MhcPreTilingData *>(tilingGm_);
            if constexpr (!ONE_POSITION) {
                positionCount_ = tilingPtr->positionCount;
            }
            if constexpr (!EPS_DEFAULT) {
                normEps_ = tilingPtr->normEps;
                hcEps_ = tilingPtr->hcEps;
            }
        }
        SyncMte2ToVector();
        if (!hasGamma_) {
            Duplicate(gammaFloat_, 1.0F, kWidth);
        }
        Duplicate(reduceOut_, 0.0F, 8U);
        // postScale_ = [1 x n, 2 x n]，postEps_ = [hcEps x n, 0 x n]，
        // 与 [pPre | pPost] 的 lane 顺序对应；用受 mask 限制的写入构造前半段。
        Duplicate(postScale_, 2.0F, 8U);
        Duplicate(postEps_, 0.0F, 8U);
        SetMaskNorm();
        SetVectorMask<float>(static_cast<int32_t>(kN));
        Adds<float, false>(postScale_, postScale_, -1.0F, 1U);
        Adds<float, false>(postEps_, postEps_, hcEps_, 1U);
        // sigmoid 用 1/(1+exp(-z))：pPre/pPost 两段连续，bias 一次取反即可。
        SetVectorMask<float>(static_cast<int32_t>(2U * kN));
        Muls<float, false>(biasFloat_, biasFloat_, -1.0F, 1U);
        // alpha 折进 phi：每位置因此不再需要任何 alpha 运算。
        SetVectorMask<float>(64);
        const UnaryRepeatParams rowStrides{1, 1, 8, 8};
        Muls<float, false>(phiFloat_, phiFloat_, -alpha0_, MASK_PLACEHOLDER,
                           static_cast<uint8_t>(kN * kWidthRepeats), rowStrides);
        Muls<float, false>(phiFloat_[kN * kWidth], phiFloat_[kN * kWidth], -alpha1_,
                           MASK_PLACEHOLDER, static_cast<uint8_t>(kN * kWidthRepeats),
                           rowStrides);
        Muls<float, false>(phiFloat_[2U * kN * kWidth], phiFloat_[2U * kN * kWidth],
                           alpha2_, MASK_PLACEHOLDER,
                           static_cast<uint8_t>(kN * kN * kWidthRepeats), rowStrides);
    }

    __aicore__ inline void IssueXLoad(uint64_t position)
    {
        const DataCopyExtParams xCopy{1, kWidth * sizeof(DT_X), 0, 0, 0};
        if constexpr (kRawX) {
            DataCopyPad(xRaw_, xGm_[position * kWidth], xCopy, {});
        } else {
            DataCopyPad(xFloat_, xGm_[position * kWidth], xCopy, {});
        }
    }

    __aicore__ inline void ProcessLoaded(uint64_t position)
    {
        const BinaryRepeatParams contiguousBinary{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams contiguousUnary{1, 1, 8, 8};
        SetVectorMask<float>(64);
        if constexpr (kRawX) {
            Cast<float, DT_X, false>(xFloat_, xRaw_, RoundMode::CAST_NONE,
                                     MASK_PLACEHOLDER,
                                     static_cast<uint8_t>(kWidthRepeats), {1, 1, 8, 4});
        }

        // invRms 用原始 xFlat 的平方均值，与 gamma 无关。
        Mul<float, false>(product_, xFloat_, xFloat_, MASK_PLACEHOLDER, 1U,
                          contiguousBinary);
        if constexpr (kWidthRepeats == 2U) {
            MulAddDst<float, float, false>(product_, xFloat_[64U], xFloat_[64U],
                                           MASK_PLACEHOLDER, 1U, contiguousBinary);
        }
        RepeatReduceSum<float, false>(reduceOut_, product_, 1, MASK_PLACEHOLDER, 0, 1, 1,
                                      0);

        // gamma 折进 x 而不是 phi：每位置只需 1~2 个 repeat。
        // 缺省 gamma 在前处理阶段被填成全 1，这里因此没有分支。
        Mul<float, false>(hInAcc_, xFloat_, gammaFloat_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kWidthRepeats), contiguousBinary);

        const BinaryRepeatParams projectStrides{
            1, 1, 1, 8, 0, static_cast<uint8_t>(8U * kWidthRepeats)};
        Mul<float, false>(product_, hInAcc_, phiFloat_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRows), projectStrides);
        if constexpr (kWidthRepeats == 2U) {
            MulAddDst<float, float, false>(product_, hInAcc_[64U], phiFloat_[64U],
                                           MASK_PLACEHOLDER,
                                           static_cast<uint8_t>(kRows), projectStrides);
        }
        RepeatReduceSum<float, false>(rowAcc_, product_, static_cast<int32_t>(kRows),
                                      MASK_PLACEHOLDER, 0, 1, 1, 8);

        SetVectorMask<float>(1);
        Muls<float, false>(reduceOut_, reduceOut_, kInvWidth, 1U);
        Adds<float, false>(reduceOut_, reduceOut_, normEps_, 1U);
        Rsqrt<float, false>(reduceOut_, reduceOut_, MASK_PLACEHOLDER, 1U,
                            contiguousUnary);
        Brcb(product_, reduceOut_, 1U, {1, 8});

        SetVectorMask<float>(8);
        const BinaryRepeatParams broadcastStrides{1, 1, 1, 1, 1, 0};
        Mul<float, false>(rowAcc_, rowAcc_, product_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRowBlocks), broadcastStrides);
        // alpha 已在前处理阶段折进 phi，这里只剩一次整段 bias 相加。
        SetVectorMask<float>(static_cast<int32_t>(kRows));
        Add<float, false>(rowAcc_, rowAcc_, biasFloat_, MASK_PLACEHOLDER, 1U,
                          contiguousBinary);

        // pPre 与 pPost 同处一个 block，sigmoid 一次覆盖两段；
        // hcEps 与 hPost 的 2 倍则由两个常量向量按 lane 施加。
        SetVectorMask<float>(static_cast<int32_t>(2U * kN));
        Exp<float, false>(sigmoidTmp_, rowAcc_, MASK_PLACEHOLDER, 1U, contiguousUnary);
        Adds<float, false>(sigmoidTmp_, sigmoidTmp_, 1.0F, 1U);
        Reciprocal<float, false>(rowAcc_, sigmoidTmp_, MASK_PLACEHOLDER, 1U,
                                 contiguousUnary);
        Mul<float, false>(rowAcc_, rowAcc_, postScale_, MASK_PLACEHOLDER, 1U,
                          contiguousBinary);
        Add<float, false>(rowAcc_, rowAcc_, postEps_, MASK_PLACEHOLDER, 1U,
                          contiguousBinary);

        // Brcb 把 pPre/pPost 每个 lane 展开成一个 block：低 kN 个 block 供 hIn 使用，
        // 紧接的 kN 个 block 即是 hPost，用跨步搬运直接写回，无需再对齐一份副本。
        Brcb(product_, rowAcc_, 1U, {1, 8});
        // 先把 hPost/hRes 发出去，其 MTE3 时延与下面的 hIn 计算重叠。
        SyncVectorToMte3();
        const DataCopyExtParams postCopy{static_cast<uint16_t>(kN), sizeof(DT_PARAM), 0,
                                         0, 0};
        const DataCopyExtParams resStoreCopy{1, kN * kN * sizeof(DT_PARAM), 0, 0, 0};
        DataCopyPad(hPostGm_[position * kN], product_[kPostLocal * 8U], postCopy);
        DataCopyPad(hResGm_[position * kN * kN], rowAcc_[kResLocal], resStoreCopy);

        SetVectorMask<float>(8);
        Mul<float, false>(hInAcc_, xFloat_, product_, MASK_PLACEHOLDER, 2U,
                          broadcastStrides);
#pragma unroll
        for (uint32_t lane = 1U; lane < kN; ++lane) {
            MulAddDst<float, float, false>(hInAcc_, xFloat_[lane * kD],
                                           product_[lane * 8U], MASK_PLACEHOLDER, 2U,
                                           broadcastStrides);
        }

        if constexpr (kRawX) {
            SetVectorMask<float>(static_cast<int32_t>(kD));
            Cast<DT_X, float, false>(xOutRaw_, hInAcc_, RoundMode::CAST_RINT,
                                     MASK_PLACEHOLDER, 1U, {1, 1, 4, 8});
        }
        SyncVectorToMte3();
        const DataCopyExtParams hInCopy{1, kD * sizeof(DT_X), 0, 0, 0};
        if constexpr (kRawX) {
            DataCopyPad(hInGm_[position * kD], xOutRaw_, hInCopy);
        } else {
            DataCopyPad(hInGm_[position * kD], hInAcc_, hInCopy);
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_PARAM> phiGm_;
    GlobalTensor<DT_PARAM> alphaGm_;
    GlobalTensor<DT_PARAM> biasGm_;
    GlobalTensor<DT_PARAM> gammaGm_;
    GlobalTensor<DT_X> hInGm_;
    GlobalTensor<DT_PARAM> hPostGm_;
    GlobalTensor<DT_PARAM> hResGm_;

    LocalTensor<DT_X> xRaw_;
    LocalTensor<DT_X> xOutRaw_;
    LocalTensor<float> xFloat_;
    LocalTensor<float> phiFloat_;
    LocalTensor<float> biasFloat_;
    LocalTensor<float> gammaFloat_;
    LocalTensor<float> product_;
    LocalTensor<float> hInAcc_;
    LocalTensor<float> reduceOut_;
    LocalTensor<float> sigmoidTmp_;
    LocalTensor<float> postScale_;
    LocalTensor<float> postEps_;
    LocalTensor<float> rowAcc_;

    GM_ADDR tilingGm_ = nullptr;
    uint64_t positionCount_ = 0;
    bool hasGamma_ = false;
    float normEps_ = 0.0F;
    float hcEps_ = 0.0F;
    float alpha0_ = 0.0F;
    float alpha1_ = 0.0F;
    float alpha2_ = 0.0F;
};

// n=8 的历史最优官方路径（官方 3.32μs）。指令序列保持原样，
// 只把原先硬编码的 eps/gamma/positionCount 改为显式传入，保证通用正确性。
template <class DT_X, class DT_PARAM, uint32_t STATIC_N, bool ONE_POSITION>
class KernelMhcPreLegacyN8 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                uint64_t positionCount, bool hasGamma, float normEps,
                                float hcEps)
    {
        positionCount_ = positionCount;
        hasGamma_ = hasGamma;
        invWidth_ = kInvWidth;
        normEps_ = normEps;
        hcEps_ = hcEps;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(bias));
        if (hasGamma_) {
            gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(gamma));
        }
        hInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(hIn));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hPost));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hRes));

        xRaw_ = LocalTensor<DT_X>(TPosition::VECIN, kXRawAddress, kWidth);
        paramRaw_ = LocalTensor<DT_PARAM>(TPosition::VECIN, kParamRawAddress, kParamLength);
        xFloat_ = LocalTensor<float>(TPosition::VECIN, kXFloatAddress, kWidth);
        paramFloat_ = LocalTensor<float>(TPosition::VECIN, kParamFloatAddress, kParamLength);
        product_ = LocalTensor<float>(TPosition::VECCALC, kProductAddress, kProductLength);
        hInAcc_ = LocalTensor<float>(TPosition::VECOUT, kHInAddress, kWidth);
        reduceOut_ = LocalTensor<float>(TPosition::VECCALC, kReduceAddress, 8U);
        scalarIn_ = LocalTensor<float>(TPosition::VECOUT, kScalarInAddress, 8U);
        scalarOut_ = LocalTensor<float>(TPosition::VECOUT, kScalarOutAddress, kScalarLength);
        scalarAux_ = LocalTensor<float>(TPosition::VECCALC, kScalarAuxAddress, 8U);
        rowAcc_ = LocalTensor<float>(TPosition::VECOUT, kRowAddress, kRowLength);
        gatherOffsets_ = LocalTensor<uint32_t>(TPosition::VECCALC, kGatherAddress, 64U);
        xOutRaw_ = LocalTensor<DT_X>(TPosition::VECOUT, kXOutAddress, 16U);
        paramOutRaw_ = LocalTensor<DT_PARAM>(TPosition::VECOUT, kParamOutAddress,
                                             kParamOutLength);
    }

    __aicore__ inline void InitOnePosition(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha,
                                           GM_ADDR bias, GM_ADDR gamma, GM_ADDR hIn,
                                           GM_ADDR hPost, GM_ADDR hRes, bool hasGamma,
                                           float normEps, float hcEps)
    {
        hasGamma_ = hasGamma;
        invWidth_ = kInvWidth;
        normEps_ = normEps;
        hcEps_ = hcEps;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(bias));
        if (hasGamma_) {
            gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(gamma));
        }
        hInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(hIn));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hPost));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hRes));

        if constexpr (!IsSameType<DT_X, float>::value) {
            xRaw_ = LocalTensor<DT_X>(TPosition::VECIN, kXRawAddress, kWidth);
        }
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            paramRaw_ = LocalTensor<DT_PARAM>(TPosition::VECIN, kParamRawAddress,
                                              kParamLength);
        }
        xFloat_ = LocalTensor<float>(TPosition::VECIN, kXFloatAddress, kWidth);
        paramFloat_ = LocalTensor<float>(TPosition::VECIN, kParamFloatAddress,
                                         kParamLength);
        product_ = LocalTensor<float>(TPosition::VECCALC, kProductAddress,
                                      kProductLength);
        hInAcc_ = LocalTensor<float>(TPosition::VECOUT, kHInAddress, kWidth);
        reduceOut_ = LocalTensor<float>(TPosition::VECCALC, kReduceAddress, 8U);
        scalarOut_ = LocalTensor<float>(TPosition::VECOUT, kScalarOutAddress,
                                        kScalarLength);
        rowAcc_ = LocalTensor<float>(TPosition::VECOUT, kRowAddress, kRowLength);
        if constexpr (!IsSameType<DT_X, float>::value) {
            xOutRaw_ = LocalTensor<DT_X>(TPosition::VECOUT, kXOutAddress, 16U);
        }
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            paramOutRaw_ = LocalTensor<DT_PARAM>(TPosition::VECOUT, kParamOutAddress,
                                                 kParamOutLength);
        }
    }

    __aicore__ inline void Process()
    {
        LoadConstants();
        if constexpr (ONE_POSITION) {
            const uint64_t position = GetBlockIdx();
            LoadX(position * kWidth);
            SetVectorMask<float>(64);
            SyncMte2ToVector();
            if constexpr (!IsSameType<DT_X, float>::value) {
                Cast<float, DT_X, false>(xFloat_, xRaw_, RoundMode::CAST_NONE,
                                         MASK_PLACEHOLDER, 2U, {1, 1, 8, 4});
                StaticVectorBarrier();
            }
            ProjectAndInvRms();
            Transform();
            if constexpr (IsSameType<DT_PARAM, float>::value) {
                SyncVectorToMte3();
                StoreParamOutputs(position);
                ComputeHIn();
                SyncVectorToMte3();
                StoreHInOutput(position);
            } else {
                ComputeHIn();
                StoreOutputs(position);
            }
            ResetMask();
        } else {
            const uint32_t blockIndex = GetBlockIdx();
            const uint32_t blockCount = GetBlockNum();
            bool waitsForPreviousStore = false;
            for (uint64_t position = blockIndex; position < positionCount_;
                 position += blockCount) {
                const bool hasNextPosition = position + blockCount < positionCount_;
                if (position != blockIndex) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                }
                LoadX(position * kWidth);
                if constexpr (kN != 4U || IsSameType<DT_X, float>::value) {
                    if (waitsForPreviousStore) {
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                }
                SetVectorMask<float>(64);
                SyncMte2ToVector();
                if constexpr (!IsSameType<DT_X, float>::value) {
                    Cast<float, DT_X, false>(xFloat_, xRaw_, RoundMode::CAST_NONE,
                                             MASK_PLACEHOLDER,
                                             static_cast<uint8_t>(kWidth / 64U),
                                             {1, 1, 8, 4});
                    StaticVectorBarrier();
                    if (hasNextPosition) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    }
                }

                ProjectAndInvRms(waitsForPreviousStore);
                Transform();
                ComputeHIn();
                if constexpr (IsSameType<DT_X, float>::value) {
                    if (hasNextPosition) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    }
                }
                StoreOutputs(position);

                waitsForPreviousStore = hasNextPosition;
                if (waitsForPreviousStore) {
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
            }
            ResetMask();
        }
    }

private:
    static constexpr uint32_t Align32(uint32_t value)
    {
        return (value + 31U) & ~31U;
    }

    __aicore__ inline void StaticVectorBarrier()
    {
    }

    static_assert(STATIC_N == 4U || STATIC_N == 8U);
    static_assert(!ONE_POSITION || STATIC_N == 8U);
    static constexpr uint32_t kN = STATIC_N;
    static constexpr bool kDirectAlpha =
        ONE_POSITION && IsSameType<DT_PARAM, float>::value;
    static constexpr uint32_t kD = 16U;
    static constexpr uint32_t kWidth = kN * kD;
    static constexpr uint32_t kRows = kN * (kN + 2U);
    static constexpr uint32_t kPhiLength = kWidth * kRows;
    static constexpr uint32_t kProductLength =
        kN == 4U ? kPhiLength : 64U * kRows;
    static constexpr float kInvWidth = 1.0F / static_cast<float>(kWidth);
    static constexpr uint32_t kBiasOffset = kPhiLength;
    static constexpr uint32_t kAlphaOffset = kN == 4U ? 1568U : 10320U;
    static constexpr uint32_t kGammaOffset = kN == 4U ? 1584U : 10336U;
    static constexpr uint32_t kParamLength = kGammaOffset + kWidth;
    static constexpr uint32_t kScalarLength = kN == 4U ? 8U : 16U;
    static constexpr uint32_t kRowLength = kN == 4U ? 32U : 80U;
    static constexpr uint32_t kParamOutLength = kN == 4U ? 32U : 80U;

    static constexpr uint32_t kXRawAddress = 0U;
    static constexpr uint32_t kParamRawAddress =
        Align32(kXRawAddress + kWidth * sizeof(DT_X));
    static constexpr uint32_t kXFloatAddress =
        Align32(kParamRawAddress + kParamLength * sizeof(DT_PARAM));
    static constexpr uint32_t kParamFloatAddress =
        Align32(kXFloatAddress + kWidth * sizeof(float));
    static constexpr uint32_t kProductAddress =
        Align32(kParamFloatAddress + kParamLength * sizeof(float));
    static constexpr uint32_t kHInAddress =
        Align32(kProductAddress + kProductLength * sizeof(float));
    static constexpr uint32_t kReduceAddress =
        Align32(kHInAddress + kWidth * sizeof(float));
    static constexpr uint32_t kScalarInAddress = kReduceAddress + 32U;
    static constexpr uint32_t kScalarOutAddress = kScalarInAddress + 32U;
    static constexpr uint32_t kScalarAuxAddress = kScalarOutAddress + 32U;
    static constexpr uint32_t kRowAddress = kScalarAuxAddress + 32U;
    static constexpr uint32_t kXOutAddress =
        Align32(kRowAddress + kRowLength * sizeof(float));
    static constexpr uint32_t kParamOutAddress =
        Align32(kXOutAddress + 16U * sizeof(DT_X));
    static constexpr uint32_t kGatherAddress =
        Align32(kParamOutAddress + kParamOutLength * sizeof(DT_PARAM));

    __aicore__ inline void SyncMte2ToVector()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToScalar()
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void SyncScalarToVector()
    {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void LoadConstants()
    {
        const DataCopyExtParams phiCopy{1, kPhiLength * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams biasCopy{1, kRows * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams alphaCopy{1, 3U * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams gammaCopy{1, kWidth * sizeof(DT_PARAM), 0, 0, 0};
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            DataCopyPad(paramFloat_, phiGm_, phiCopy, {});
            DataCopyPad(paramFloat_[kBiasOffset], biasGm_, biasCopy, {});
            if constexpr (!kDirectAlpha) {
                DataCopyPad(paramFloat_[kAlphaOffset], alphaGm_, alphaCopy, {});
            }
            if (hasGamma_) {
                DataCopyPad(paramFloat_[kGammaOffset], gammaGm_, gammaCopy, {});
            }
        } else {
            DataCopyPad(paramRaw_, phiGm_, phiCopy, {});
            DataCopyPad(paramRaw_[kBiasOffset], biasGm_, biasCopy, {});
            DataCopyPad(paramRaw_[kAlphaOffset], alphaGm_, alphaCopy, {});
            if (hasGamma_) {
                DataCopyPad(paramRaw_[kGammaOffset], gammaGm_, gammaCopy, {});
            }
        }
        if constexpr (kDirectAlpha) {
            alpha0_ = alphaGm_.GetValue(0U);
            alpha1_ = alphaGm_.GetValue(1U);
            alpha2_ = alphaGm_.GetValue(2U);
        }
        SyncMte2ToVector();
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            Cast(paramFloat_, paramRaw_, RoundMode::CAST_NONE, kParamLength);
            StaticVectorBarrier();
        }

        if constexpr (kN == 4U) {
            if (hasGamma_) {
                const BinaryRepeatParams scaleStrides{1, 1, 1, 8, 0, 8};
                Mul(paramFloat_, paramFloat_[kGammaOffset], paramFloat_,
                    static_cast<uint64_t>(kWidth), static_cast<uint8_t>(kRows),
                    scaleStrides);
                StaticVectorBarrier();
            }
        }
        if constexpr (!kDirectAlpha) {
            SyncVectorToScalar();
            alpha0_ = paramFloat_.GetValue(kAlphaOffset);
            alpha1_ = paramFloat_.GetValue(kAlphaOffset + 1U);
            alpha2_ = paramFloat_.GetValue(kAlphaOffset + 2U);
        }
        if constexpr (kN == 4U) {
#pragma unroll
            for (uint32_t index = 0; index < 4U; ++index) {
                gatherOffsets_.SetValue(index, (4U + index) * sizeof(float));
                gatherOffsets_.SetValue(4U + index, (4U + index) * sizeof(float));
            }
        }
        if constexpr (!kDirectAlpha) {
            SyncScalarToVector();
        }
        if constexpr (kN == 4U) {
            Duplicate(gatherOffsets_[8U], 0U, 56U);
        }
        Duplicate(reduceOut_, 0.0F, 8U);
        SetMaskNorm();
        if constexpr (!ONE_POSITION) {
            SetVectorMask<float>(64);
            const UnaryRepeatParams contiguousUnary{1, 1, 8, 8};
            Muls<float, false>(paramFloat_, paramFloat_, -alpha0_,
                               MASK_PLACEHOLDER,
                               static_cast<uint8_t>(kN * kWidth / 64U),
                               contiguousUnary);
            Muls<float, false>(paramFloat_[kN * kWidth],
                               paramFloat_[kN * kWidth], -alpha1_,
                               MASK_PLACEHOLDER,
                               static_cast<uint8_t>(kN * kWidth / 64U),
                               contiguousUnary);
            Muls<float, false>(paramFloat_[2U * kN * kWidth],
                               paramFloat_[2U * kN * kWidth], alpha2_,
                               MASK_PLACEHOLDER,
                               static_cast<uint8_t>(kN * kN * kWidth / 64U),
                               contiguousUnary);
        }
        SetVectorMask<float>(static_cast<int32_t>(2U * kN));
        Muls<float, false>(paramFloat_[kBiasOffset], paramFloat_[kBiasOffset],
                           -1.0F, 1U);
        StaticVectorBarrier();
    }

    __aicore__ inline void LoadX(uint64_t offset)
    {
        const DataCopyExtParams copyParams{1, kWidth * sizeof(DT_X), 0, 0, 0};
        if constexpr (IsSameType<DT_X, float>::value) {
            DataCopyPad(xFloat_, xGm_[offset], copyParams, {});
        } else {
            DataCopyPad(xRaw_, xGm_[offset], copyParams, {});
        }
    }

    __aicore__ inline void ProjectAndInvRms(bool waitsForPreviousStore = false)
    {
        const BinaryRepeatParams contiguousBinary{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams contiguousUnary{1, 1, 8, 8};
        if constexpr (kN == 4U) {
            Mul<float, false>(product_, xFloat_, xFloat_, MASK_PLACEHOLDER, 1U,
                              contiguousBinary);
            StaticVectorBarrier();
            RepeatReduceSum<float, false>(reduceOut_, product_, 1, MASK_PLACEHOLDER,
                                          0, 1, 1, 0);

            const BinaryRepeatParams projectionStrides{1, 1, 1, 8, 0, 8};
            Mul<float, false>(product_, xFloat_, paramFloat_, MASK_PLACEHOLDER,
                              static_cast<uint8_t>(kRows), projectionStrides);
            StaticVectorBarrier();
            if constexpr (!IsSameType<DT_X, float>::value) {
                if (waitsForPreviousStore) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
            }
            RepeatReduceSum<float, false>(rowAcc_, product_,
                                          static_cast<int32_t>(kRows),
                                          MASK_PLACEHOLDER, 0, 1, 1, 8);
        } else {
            Mul<float, false>(product_, xFloat_, xFloat_, MASK_PLACEHOLDER, 1U,
                              contiguousBinary);
            StaticVectorBarrier();
            MulAddDst<float, float, false>(product_, xFloat_[64U], xFloat_[64U],
                                           MASK_PLACEHOLDER, 1U, contiguousBinary);
            StaticVectorBarrier();
            RepeatReduceSum<float, false>(reduceOut_, product_, 1, MASK_PLACEHOLDER,
                                          0, 1, 1, 0);

            const BinaryRepeatParams projectionHalfStrides{1, 1, 1, 8, 0, 16};
            if (hasGamma_) {
                Mul<float, false>(hInAcc_, xFloat_, paramFloat_[kGammaOffset],
                                  MASK_PLACEHOLDER, 2U, contiguousBinary);
                StaticVectorBarrier();
                Mul<float, false>(product_, hInAcc_, paramFloat_, MASK_PLACEHOLDER,
                                  static_cast<uint8_t>(kRows),
                                  projectionHalfStrides);
                StaticVectorBarrier();
                MulAddDst<float, float, false>(product_, hInAcc_[64U],
                                               paramFloat_[64U], MASK_PLACEHOLDER,
                                               static_cast<uint8_t>(kRows),
                                               projectionHalfStrides);
            } else {
                Mul<float, false>(product_, xFloat_, paramFloat_, MASK_PLACEHOLDER,
                                  static_cast<uint8_t>(kRows),
                                  projectionHalfStrides);
                StaticVectorBarrier();
                MulAddDst<float, float, false>(product_, xFloat_[64U],
                                               paramFloat_[64U], MASK_PLACEHOLDER,
                                               static_cast<uint8_t>(kRows),
                                               projectionHalfStrides);
            }
            StaticVectorBarrier();
            RepeatReduceSum<float, false>(rowAcc_, product_,
                                          static_cast<int32_t>(kRows),
                                          MASK_PLACEHOLDER, 0, 1, 1, 8);
        }
        SetVectorMask<float>(1);
        Muls<float, false>(reduceOut_, reduceOut_, invWidth_, 1U);
        StaticVectorBarrier();
        Adds<float, false>(reduceOut_, reduceOut_, normEps_, 1U);
        StaticVectorBarrier();
        Rsqrt<float, false>(reduceOut_, reduceOut_, MASK_PLACEHOLDER, 1U,
                            contiguousUnary);
    }

    __aicore__ inline void Transform()
    {
        StaticVectorBarrier();
        Brcb(product_, reduceOut_, 1U, {1, 8});
        StaticVectorBarrier();
        SetVectorMask<float>(8);
        const BinaryRepeatParams normalizeStrides{1, 1, 1, 1, 1, 0};
        const UnaryRepeatParams contiguousUnary{1, 1, 8, 8};
        Mul<float, false>(rowAcc_, rowAcc_, product_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRows / 8U), normalizeStrides);
        StaticVectorBarrier();
        if constexpr (ONE_POSITION) {
            SetVectorMask<float>(static_cast<int32_t>(kN));
            Muls<float, false>(rowAcc_, rowAcc_, -alpha0_, MASK_PLACEHOLDER, 1U,
                               contiguousUnary);
            Muls<float, false>(rowAcc_[kN], rowAcc_[kN], -alpha1_,
                               MASK_PLACEHOLDER, 1U, contiguousUnary);
            SetVectorMask<float>(static_cast<int32_t>(kN * kN));
            Muls<float, false>(rowAcc_[2U * kN], rowAcc_[2U * kN], alpha2_,
                               MASK_PLACEHOLDER, 1U, contiguousUnary);
            StaticVectorBarrier();
        }
        if constexpr (kN == 4U) {
            SetVectorMask<float>(static_cast<int32_t>(kRows));
            const BinaryRepeatParams contiguousBinary{1, 1, 1, 8, 8, 8};
            Add<float, false>(rowAcc_, rowAcc_, paramFloat_[kBiasOffset],
                              MASK_PLACEHOLDER, 1U, contiguousBinary);
        } else {
            SetVectorMask<float>(16);
            const BinaryRepeatParams contiguousRows{1, 1, 1, 2, 2, 2};
            Add<float, false>(rowAcc_, rowAcc_, paramFloat_[kBiasOffset],
                              MASK_PLACEHOLDER, 5U, contiguousRows);
        }
        StaticVectorBarrier();

        SetVectorMask<float>(static_cast<int32_t>(2U * kN));
        Exp<float, false>(scalarOut_, rowAcc_, MASK_PLACEHOLDER, 1U,
                          contiguousUnary);
        StaticVectorBarrier();
        Adds<float, false>(scalarOut_, scalarOut_, 1.0F, 1U);
        StaticVectorBarrier();
        Reciprocal<float, false>(rowAcc_, scalarOut_, MASK_PLACEHOLDER, 1U,
                                 contiguousUnary);
        StaticVectorBarrier();
        SetVectorMask<float>(static_cast<int32_t>(kN));
        Adds<float, false>(rowAcc_, rowAcc_, hcEps_, 1U);
        StaticVectorBarrier();
        if constexpr (kN == 4U) {
            Gather(scalarOut_, rowAcc_, gatherOffsets_, 0U, 4U);
            StaticVectorBarrier();
            Muls<float, false>(scalarOut_, scalarOut_, 2.0F, 1U);
        } else {
            Muls<float, false>(rowAcc_[kN], rowAcc_[kN], 2.0F, 1U);
        }
    }

    __aicore__ inline void ComputeHIn()
    {
        StaticVectorBarrier();
        Brcb(product_, rowAcc_, 1U, {1, 8});
        StaticVectorBarrier();
        SetVectorMask<float>(8);
        const BinaryRepeatParams laneWeightStrides{1, 1, 1, 1, 1, 0};
        Mul<float, false>(hInAcc_, xFloat_, product_, MASK_PLACEHOLDER, 2U,
                          laneWeightStrides);
        StaticVectorBarrier();
        MulAddDst<float, float, false>(hInAcc_, xFloat_[16U], product_[8U],
                                      MASK_PLACEHOLDER, 2U, laneWeightStrides);
        StaticVectorBarrier();
        MulAddDst<float, float, false>(hInAcc_, xFloat_[32U], product_[16U],
                                      MASK_PLACEHOLDER, 2U, laneWeightStrides);
        StaticVectorBarrier();
        MulAddDst<float, float, false>(hInAcc_, xFloat_[48U], product_[24U],
                                      MASK_PLACEHOLDER, 2U, laneWeightStrides);
        if constexpr (kN == 8U) {
            StaticVectorBarrier();
            MulAddDst<float, float, false>(hInAcc_, xFloat_[64U], product_[32U],
                                           MASK_PLACEHOLDER, 2U,
                                           laneWeightStrides);
            StaticVectorBarrier();
            MulAddDst<float, float, false>(hInAcc_, xFloat_[80U], product_[40U],
                                           MASK_PLACEHOLDER, 2U,
                                           laneWeightStrides);
            StaticVectorBarrier();
            MulAddDst<float, float, false>(hInAcc_, xFloat_[96U], product_[48U],
                                           MASK_PLACEHOLDER, 2U,
                                           laneWeightStrides);
            StaticVectorBarrier();
            MulAddDst<float, float, false>(hInAcc_, xFloat_[112U], product_[56U],
                                           MASK_PLACEHOLDER, 2U,
                                           laneWeightStrides);
        }

        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            if constexpr (kN == 4U) {
                SetVectorMask<float>(4);
                Cast<DT_PARAM, float, false>(paramOutRaw_, scalarOut_,
                                             RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                                             1U, {1, 1, 4, 8});
                SetVectorMask<float>(16);
                Cast<DT_PARAM, float, false>(paramOutRaw_[16U], rowAcc_[8U],
                                             RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                                             1U, {1, 1, 4, 8});
            } else {
                SetVectorMask<float>(8);
                Cast<DT_PARAM, float, false>(paramOutRaw_, rowAcc_[kN],
                                             RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                                             1U, {1, 1, 4, 8});
                SetVectorMask<float>(64);
                Cast<DT_PARAM, float, false>(paramOutRaw_[16U], rowAcc_[2U * kN],
                                             RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                                             1U, {1, 1, 4, 8});
            }
        }
        if constexpr (!IsSameType<DT_X, float>::value) {
            SetVectorMask<float>(16);
            Cast<DT_X, float, false>(xOutRaw_, hInAcc_, RoundMode::CAST_RINT,
                                     MASK_PLACEHOLDER, 1U, {1, 1, 4, 8});
        }
    }

    __aicore__ inline void StoreParamOutputs(uint64_t position)
    {
        const DataCopyExtParams postCopy{1, kN * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams resCopy{1, kN * kN * sizeof(DT_PARAM), 0, 0, 0};
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            if constexpr (kN == 4U) {
                DataCopyPad(hPostGm_[position * kN], scalarOut_, postCopy);
                DataCopyPad(hResGm_[position * kN * kN], rowAcc_[2U * kN],
                            resCopy);
            } else {
                DataCopyPad(hPostGm_[position * kN], rowAcc_[kN], postCopy);
                DataCopyPad(hResGm_[position * kN * kN], rowAcc_[2U * kN],
                            resCopy);
            }
        } else {
            DataCopyPad(hPostGm_[position * kN], paramOutRaw_, postCopy);
            DataCopyPad(hResGm_[position * kN * kN], paramOutRaw_[16U], resCopy);
        }
    }

    __aicore__ inline void StoreHInOutput(uint64_t position)
    {
        const DataCopyExtParams hInCopy{1, 16U * sizeof(DT_X), 0, 0, 0};
        if constexpr (IsSameType<DT_X, float>::value) {
            DataCopyPad(hInGm_[position * 16U], hInAcc_, hInCopy);
        } else {
            DataCopyPad(hInGm_[position * 16U], xOutRaw_, hInCopy);
        }
    }

    __aicore__ inline void StoreOutputs(uint64_t position)
    {
        SyncVectorToMte3();
        StoreParamOutputs(position);
        StoreHInOutput(position);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_PARAM> phiGm_;
    GlobalTensor<DT_PARAM> alphaGm_;
    GlobalTensor<DT_PARAM> biasGm_;
    GlobalTensor<DT_PARAM> gammaGm_;
    GlobalTensor<DT_X> hInGm_;
    GlobalTensor<DT_PARAM> hPostGm_;
    GlobalTensor<DT_PARAM> hResGm_;

    LocalTensor<DT_X> xRaw_;
    LocalTensor<DT_PARAM> paramRaw_;
    LocalTensor<float> xFloat_;
    LocalTensor<float> paramFloat_;
    LocalTensor<float> product_;
    LocalTensor<float> hInAcc_;
    LocalTensor<float> reduceOut_;
    LocalTensor<float> scalarIn_;
    LocalTensor<float> scalarOut_;
    LocalTensor<float> scalarAux_;
    LocalTensor<float> rowAcc_;
    LocalTensor<uint32_t> gatherOffsets_;
    LocalTensor<DT_X> xOutRaw_;
    LocalTensor<DT_PARAM> paramOutRaw_;

    uint64_t positionCount_ = 0;
    bool hasGamma_ = false;
    float invWidth_ = 0.0F;
    float normEps_ = 0.0F;
    float hcEps_ = 0.0F;
    float alpha0_ = 0.0F;
    float alpha1_ = 0.0F;
    float alpha2_ = 0.0F;
};

template <class DT_X, class DT_PARAM>
class KernelMhcPre {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                const MhcPreTilingData &tiling)
    {
        positionCount_ = tiling.positionCount;
        n_ = tiling.n;
        d_ = tiling.d;
        width_ = tiling.width;
        rowCount_ = tiling.rowCount;
        tileLength_ = tiling.tileLength;
        postLocalOffset_ = ((n_ + 7U) / 8U) * 8U;
        resLocalOffset_ = postLocalOffset_ + ((n_ + 7U) / 8U) * 8U;
        hasGamma_ = tiling.hasGamma != 0;
        invWidth_ = tiling.invWidth;
        normEps_ = tiling.normEps;
        hcEps_ = tiling.hcEps;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(bias));
        if (hasGamma_) {
            gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(gamma));
        }
        hInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(hIn));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hPost));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_PARAM *>(hRes));

        const uint32_t rowBufferLength = resLocalOffset_ + n_ * n_;
        const uint32_t rowBytes = ((rowBufferLength * sizeof(float) + 31U) / 32U) * 32U;
        const uint32_t xRawBytes = tileLength_ * sizeof(DT_X);
        const uint32_t paramRawBytes = tileLength_ * sizeof(DT_PARAM);
        const uint32_t floatBytes = tileLength_ * sizeof(float);
        const uint32_t scratchBytes = xRawBytes + paramRawBytes + 5U * floatBytes +
                                      5U * 32U + rowBytes;
        pipe_.InitBuffer(scratchBuf_, scratchBytes);

        uint32_t offset = 0;
        xRaw_ = scratchBuf_.GetWithOffset<DT_X>(tileLength_, offset);
        offset += xRawBytes;
        paramRaw_ = scratchBuf_.GetWithOffset<DT_PARAM>(tileLength_, offset);
        offset += paramRawBytes;
        xFloat_ = scratchBuf_.GetWithOffset<float>(tileLength_, offset);
        offset += floatBytes;
        paramFloat_ = scratchBuf_.GetWithOffset<float>(tileLength_, offset);
        offset += floatBytes;
        xGamma_ = scratchBuf_.GetWithOffset<float>(tileLength_, offset);
        offset += floatBytes;
        product_ = scratchBuf_.GetWithOffset<float>(tileLength_, offset);
        offset += floatBytes;
        hInAcc_ = scratchBuf_.GetWithOffset<float>(tileLength_, offset);
        offset += floatBytes;
        reduceTmp_ = scratchBuf_.GetWithOffset<float>(8U, offset);
        offset += 32U;
        reduceOut_ = scratchBuf_.GetWithOffset<float>(8U, offset);
        offset += 32U;
        scalarIn_ = scratchBuf_.GetWithOffset<float>(8U, offset);
        offset += 32U;
        scalarOut_ = scratchBuf_.GetWithOffset<float>(8U, offset);
        offset += 32U;
        scalarAux_ = scratchBuf_.GetWithOffset<float>(8U, offset);
        offset += 32U;
        rowAcc_ = scratchBuf_.GetWithOffset<float>(rowBytes / sizeof(float), offset);
    }

    __aicore__ inline void Process()
    {
        alpha0_ = ReadParam(alphaGm_, 0);
        alpha1_ = ReadParam(alphaGm_, 1);
        alpha2_ = ReadParam(alphaGm_, 2);
        const uint32_t blockIndex = GetBlockIdx();
        const uint32_t blockCount = GetBlockNum();
        for (uint64_t position = blockIndex; position < positionCount_; position += blockCount) {
            ProcessPosition(position);
        }
    }

private:
    __aicore__ inline void SyncMte2ToVector()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void SyncVectorToScalar()
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void SyncScalarToVector()
    {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncScalarToMte3()
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void SyncMte3ToVector()
    {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void WaitStore()
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline float ReadParam(const GlobalTensor<DT_PARAM> &tensor, uint64_t index) const
    {
        return static_cast<float>(tensor.GetValue(index));
    }

    __aicore__ inline void LoadXFloat(uint64_t offset, uint32_t count)
    {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        if constexpr (IsSameType<DT_X, float>::value) {
            DataCopyPad(xFloat_, xGm_[offset], copyParams, {});
        } else {
            DataCopyPad(xRaw_, xGm_[offset], copyParams, {});
        }
        SyncMte2ToVector();
        if constexpr (!IsSameType<DT_X, float>::value) {
            Cast(xFloat_, xRaw_, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            SyncVectorToMte2();
        }
    }

    __aicore__ inline void LoadParamFloat(const GlobalTensor<DT_PARAM> &source,
                                           uint64_t offset, uint32_t count)
    {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(DT_PARAM)), 0, 0, 0};
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            DataCopyPad(paramFloat_, source[offset], copyParams, {});
        } else {
            DataCopyPad(paramRaw_, source[offset], copyParams, {});
        }
        SyncMte2ToVector();
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            Cast(paramFloat_, paramRaw_, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            SyncVectorToMte2();
        }
    }

    __aicore__ inline float ReduceProduct(uint32_t count)
    {
        ReduceSum(reduceOut_, product_, reduceTmp_, static_cast<int32_t>(count));
        SyncVectorToScalar();
        return reduceOut_.GetValue(0);
    }

    __aicore__ inline float ComputeInvRms(float squareSum)
    {
        const float meanWithEps = squareSum * invWidth_ + normEps_;
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            scalarIn_.SetValue(i, meanWithEps);
            scalarAux_.SetValue(i, 1.0F);
        }
        SyncScalarToVector();
        Sqrt(scalarOut_, scalarIn_, 8);
        PipeBarrier<PIPE_V>();
        Div(scalarIn_, scalarAux_, scalarOut_, 8);
        SyncVectorToScalar();
        return scalarIn_.GetValue(0);
    }

    __aicore__ inline float Sigmoid(float value)
    {
        const float expInput = value >= 0.0F ? -value : value;
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            scalarIn_.SetValue(i, expInput);
        }
        SyncScalarToVector();
        Exp(scalarOut_, scalarIn_, 8);
        SyncVectorToScalar();
        const float expValue = scalarOut_.GetValue(0);
        if (value >= 0.0F) {
            return 1.0F / (1.0F + expValue);
        }
        return expValue / (1.0F + expValue);
    }

    __aicore__ inline float Project(uint64_t xOffset)
    {
        for (uint32_t row = 0; row < rowCount_; ++row) {
            rowAcc_.SetValue(row, 0.0F);
        }

        float squareSum = 0.0F;
        for (uint32_t offset = 0; offset < width_; offset += tileLength_) {
            const uint32_t remaining = width_ - offset;
            const uint32_t count = tileLength_ < remaining ? tileLength_ : remaining;
            LoadXFloat(xOffset + offset, count);

            Mul(product_, xFloat_, xFloat_, count);
            PipeBarrier<PIPE_V>();
            squareSum += ReduceProduct(count);

            if (hasGamma_) {
                LoadParamFloat(gammaGm_, offset, count);
                Mul(xGamma_, xFloat_, paramFloat_, count);
                PipeBarrier<PIPE_V>();
            } else {
                Adds(xGamma_, xFloat_, 0.0F, count);
                PipeBarrier<PIPE_V>();
            }
            SyncVectorToMte2();

            for (uint32_t row = 0; row < rowCount_; ++row) {
                const uint64_t phiOffset = static_cast<uint64_t>(row) * width_ + offset;
                LoadParamFloat(phiGm_, phiOffset, count);
                Mul(product_, xGamma_, paramFloat_, count);
                PipeBarrier<PIPE_V>();
                SyncVectorToMte2();
                const float partial = ReduceProduct(count);
                rowAcc_.SetValue(row, rowAcc_.GetValue(row) + partial);
            }
        }
        return ComputeInvRms(squareSum);
    }

    __aicore__ inline void TransformAndStore(uint64_t position, float invRms)
    {
        for (uint32_t remainingRows = rowCount_; remainingRows > 0; --remainingRows) {
            const uint32_t row = remainingRows - 1U;
            const float projected = rowAcc_.GetValue(row) * invRms;
            const float bias = ReadParam(biasGm_, row);
            if (row < n_) {
                const float hPre = Sigmoid(projected * alpha0_ + bias) + hcEps_;
                rowAcc_.SetValue(row, hPre);
            } else if (row < 2U * n_) {
                const float hPost = 2.0F * Sigmoid(projected * alpha1_ + bias);
                rowAcc_.SetValue(postLocalOffset_ + row - n_, hPost);
            } else {
                const float hRes = projected * alpha2_ + bias;
                rowAcc_.SetValue(resLocalOffset_ + row - 2U * n_, hRes);
            }
        }
        StoreParamRange(hPostGm_, position * n_, postLocalOffset_, n_);
        StoreParamRange(hResGm_, position * static_cast<uint64_t>(n_) * n_,
                        resLocalOffset_, n_ * n_);
    }

    __aicore__ inline void StoreParamRange(GlobalTensor<DT_PARAM> &destination,
                                            uint64_t destinationOffset,
                                            uint32_t sourceOffset, uint32_t count)
    {
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            SyncScalarToMte3();
        } else {
            SyncScalarToVector();
        }
        for (uint32_t offset = 0; offset < count; offset += tileLength_) {
            const uint32_t remaining = count - offset;
            const uint32_t current = tileLength_ < remaining ? tileLength_ : remaining;
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(current * sizeof(DT_PARAM)), 0, 0, 0};
            if constexpr (IsSameType<DT_PARAM, float>::value) {
                DataCopyPad(destination[destinationOffset + offset],
                            rowAcc_[sourceOffset + offset], copyParams);
                WaitStore();
                if (offset + current < count) {
                    SyncScalarToMte3();
                }
            } else {
                Cast(paramRaw_, rowAcc_[sourceOffset + offset], RoundMode::CAST_RINT, current);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(destination[destinationOffset + offset], paramRaw_, copyParams);
                SyncMte3ToVector();
            }
        }
    }

    __aicore__ inline void StoreHIn(uint64_t offset, uint32_t count)
    {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        if constexpr (IsSameType<DT_X, float>::value) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(hInGm_[offset], hInAcc_, copyParams);
            SyncMte3ToVector();
        } else {
            Cast(xRaw_, hInAcc_, RoundMode::CAST_RINT, count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(hInGm_[offset], xRaw_, copyParams);
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void ComputeHIn(uint64_t position, uint64_t xOffset)
    {
        const uint64_t hInOffset = position * d_;
        for (uint32_t dOffset = 0; dOffset < d_; dOffset += tileLength_) {
            const uint32_t remaining = d_ - dOffset;
            const uint32_t count = tileLength_ < remaining ? tileLength_ : remaining;
            Duplicate(hInAcc_, 0.0F, count);
            PipeBarrier<PIPE_V>();
            for (uint32_t lane = 0; lane < n_; ++lane) {
                const uint64_t laneOffset = xOffset + static_cast<uint64_t>(lane) * d_ + dOffset;
                LoadXFloat(laneOffset, count);
                Muls(product_, xFloat_, rowAcc_.GetValue(lane), count);
                PipeBarrier<PIPE_V>();
                SyncVectorToMte2();
                Add(hInAcc_, hInAcc_, product_, count);
                PipeBarrier<PIPE_V>();
            }
            StoreHIn(hInOffset + dOffset, count);
        }
    }

    __aicore__ inline void ProcessPosition(uint64_t position)
    {
        const uint64_t xOffset = position * width_;
        const float invRms = Project(xOffset);
        TransformAndStore(position, invRms);
        ComputeHIn(position, xOffset);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> scratchBuf_;

    LocalTensor<DT_X> xRaw_;
    LocalTensor<DT_PARAM> paramRaw_;
    LocalTensor<float> xFloat_;
    LocalTensor<float> paramFloat_;
    LocalTensor<float> xGamma_;
    LocalTensor<float> product_;
    LocalTensor<float> hInAcc_;
    LocalTensor<float> reduceTmp_;
    LocalTensor<float> reduceOut_;
    LocalTensor<float> scalarIn_;
    LocalTensor<float> scalarOut_;
    LocalTensor<float> scalarAux_;
    LocalTensor<float> rowAcc_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_PARAM> phiGm_;
    GlobalTensor<DT_PARAM> alphaGm_;
    GlobalTensor<DT_PARAM> biasGm_;
    GlobalTensor<DT_PARAM> gammaGm_;
    GlobalTensor<DT_X> hInGm_;
    GlobalTensor<DT_PARAM> hPostGm_;
    GlobalTensor<DT_PARAM> hResGm_;

    uint64_t positionCount_ = 0;
    uint32_t n_ = 0;
    uint32_t d_ = 0;
    uint32_t width_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t postLocalOffset_ = 0;
    uint32_t resLocalOffset_ = 0;
    uint32_t tileLength_ = 0;
    bool hasGamma_ = false;
    float invWidth_ = 0.0F;
    float normEps_ = 0.0F;
    float hcEps_ = 0.0F;
    float alpha0_ = 0.0F;
    float alpha1_ = 0.0F;
    float alpha2_ = 0.0F;
};

template <typename DT_X, typename DT_PARAM, int FAST_SMALL, int FAST_ONE_POSITION,
          int FAST_N8, int EPS_DEFAULT>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                   GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    if constexpr (FAST_SMALL != 0 && IsSameType<DT_PARAM, float>::value) {
        InitSocState();
        if constexpr (FAST_N8 != 0) {
            // n=8：沿用官方实测最优的历史路径（3.32μs），新版分组排布在该形状上没有收益。
            const bool hasGamma = gamma != nullptr;
            float normEps = 1e-6F;
            float hcEps = 1e-6F;
            uint64_t positionCount = 0;
            if constexpr (EPS_DEFAULT == 0 || FAST_ONE_POSITION == 0) {
                __gm__ MhcPreTilingData *tilingPtr =
                    reinterpret_cast<__gm__ MhcPreTilingData *>(tiling);
                if constexpr (FAST_ONE_POSITION == 0) {
                    positionCount = tilingPtr->positionCount;
                }
                if constexpr (EPS_DEFAULT == 0) {
                    normEps = tilingPtr->normEps;
                    hcEps = tilingPtr->hcEps;
                }
            }
            KernelMhcPreLegacyN8<DT_X, DT_PARAM, 8U, FAST_ONE_POSITION != 0> op;
            if constexpr (FAST_ONE_POSITION != 0) {
                op.InitOnePosition(x, phi, alpha, bias, gamma, hIn, hPost, hRes,
                                   hasGamma, normEps, hcEps);
            } else {
                op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, positionCount,
                        hasGamma, normEps, hcEps);
            }
            op.Process();
        } else {
            // n=4：分组排布 + 前处理访存重叠，官方从 5.48μs 降到 4.12μs。
            KernelMhcPreSmallStatic<DT_X, DT_PARAM, 4U, FAST_ONE_POSITION != 0,
                                    EPS_DEFAULT != 0> op;
            op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tiling);
            op.Process();
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tilingData, tiling);
        KernelMhcPre<DT_X, DT_PARAM> op;
        op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
        op.Process();
    }
}
