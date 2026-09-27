#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

// 静态形状快速路径：D=16 且 n 为 4/8 时展开为编译期常量，
// 行缓冲按 32 字节分组排布，使 pPre/pPost/pRes 三段全部对齐，
// 从而避免 Gather 与标量-向量往返同步。
template <class DT_X, class DT_PARAM, uint32_t STATIC_N, bool ONE_POSITION>
class KernelMhcPreSmallStatic {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                const MhcPreTilingData &tiling)
    {
        positionCount_ = tiling.positionCount;
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

        if constexpr (!IsSameType<DT_X, float>::value) {
            xRaw_ = LocalTensor<DT_X>(TPosition::VECIN, kXRawAddress, kWidth);
            xOutRaw_ = LocalTensor<DT_X>(TPosition::VECOUT, kXOutAddress, kD);
        }
        xFloat_ = LocalTensor<float>(TPosition::VECIN, kXFloatAddress, kWidth);
        phiFloat_ = LocalTensor<float>(TPosition::VECIN, kPhiFloatAddress, kPhiLength);
        biasFloat_ = LocalTensor<float>(TPosition::VECIN, kBiasFloatAddress, kRowLength);
        gammaFloat_ = LocalTensor<float>(TPosition::VECIN, kGammaFloatAddress, kWidth);
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            phiRaw_ = LocalTensor<DT_PARAM>(TPosition::VECIN, kPhiRawAddress, kPhiLength);
            biasRaw_ = LocalTensor<DT_PARAM>(TPosition::VECIN, kBiasRawAddress,
                                             kBiasRawLength);
            gammaRaw_ = LocalTensor<DT_PARAM>(TPosition::VECIN, kGammaRawAddress, kWidth);
            paramOutRaw_ = LocalTensor<DT_PARAM>(TPosition::VECOUT, kParamOutAddress,
                                                 kParamOutLength);
        }
        product_ = LocalTensor<float>(TPosition::VECCALC, kProductAddress, kProductLength);
        hInAcc_ = LocalTensor<float>(TPosition::VECOUT, kHInAddress, kWidth);
        reduceOut_ = LocalTensor<float>(TPosition::VECCALC, kReduceAddress, 8U);
        sigmoidTmp_ = LocalTensor<float>(TPosition::VECCALC, kSigmoidAddress, 16U);
        rowAcc_ = LocalTensor<float>(TPosition::VECOUT, kRowAddress, kRowLength);
    }

    __aicore__ inline void Process()
    {
        LoadConstants();
        if constexpr (ONE_POSITION) {
            ProcessPosition(GetBlockIdx());
        } else {
            const uint64_t blockIndex = GetBlockIdx();
            const uint64_t blockCount = GetBlockNum();
            for (uint64_t position = blockIndex; position < positionCount_;
                 position += blockCount) {
                if (position != blockIndex) {
                    SyncMte3ToVector();
                    SyncVectorToMte2();
                }
                ProcessPosition(position);
            }
        }
        ResetMask();
    }

private:
    static_assert(STATIC_N == 4U || STATIC_N == 8U);
    static constexpr uint32_t kN = STATIC_N;
    static constexpr uint32_t kD = 16U;
    static constexpr uint32_t kWidth = kN * kD;
    static constexpr uint32_t kRows = kN * (kN + 2U);
    static constexpr uint32_t kPhiLength = kWidth * kRows;
    static constexpr uint32_t kWidthRepeats = kWidth / 64U;
    static constexpr uint32_t kProductLength = 64U * kRows;
    // 行缓冲分组偏移：三段起点均为 32 字节整数倍，n=8 时恰好连续。
    static constexpr uint32_t kPostLocal = 8U;
    static constexpr uint32_t kResLocal = 16U;
    static constexpr uint32_t kRowLength = kResLocal + kN * kN;
    // DT_PARAM 原始缓冲按 16 元素分组，fp16/fp32 下起点都满足 32 字节对齐。
    static constexpr uint32_t kBiasRawPost = 16U;
    static constexpr uint32_t kBiasRawRes = 32U;
    static constexpr uint32_t kBiasRawLength = kBiasRawRes + kN * kN;
    static constexpr uint32_t kParamOutLength = 16U + kN * kN;
    static constexpr bool kRawX = !IsSameType<DT_X, float>::value;
    static constexpr bool kRawParam = !IsSameType<DT_PARAM, float>::value;

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
    static constexpr uint32_t kPhiRawAddress = Align32(kGammaFloatAddress + kWidth * 4U);
    static constexpr uint32_t kBiasRawAddress =
        Align32(kPhiRawAddress + (kRawParam ? kPhiLength * sizeof(DT_PARAM) : 0U));
    static constexpr uint32_t kGammaRawAddress =
        Align32(kBiasRawAddress + (kRawParam ? kBiasRawLength * sizeof(DT_PARAM) : 0U));
    static constexpr uint32_t kProductAddress =
        Align32(kGammaRawAddress + (kRawParam ? kWidth * sizeof(DT_PARAM) : 0U));
    static constexpr uint32_t kHInAddress = Align32(kProductAddress + kProductLength * 4U);
    static constexpr uint32_t kReduceAddress = Align32(kHInAddress + kWidth * 4U);
    static constexpr uint32_t kSigmoidAddress = kReduceAddress + 32U;
    static constexpr uint32_t kRowAddress = Align32(kSigmoidAddress + 16U * 4U);
    static constexpr uint32_t kXOutAddress = Align32(kRowAddress + kRowLength * 4U);
    static constexpr uint32_t kParamOutAddress =
        Align32(kXOutAddress + (kRawX ? kD * sizeof(DT_X) : 0U));

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
        // alpha 直接用标量读 GM，避免向量-标量同步屏障。
        alpha0_ = static_cast<float>(alphaGm_.GetValue(0U));
        alpha1_ = static_cast<float>(alphaGm_.GetValue(1U));
        alpha2_ = static_cast<float>(alphaGm_.GetValue(2U));

        const DataCopyExtParams phiCopy{1, kPhiLength * sizeof(DT_PARAM), 0, 0, 0};
        // 两个 block 各自补齐到 32 字节落位，使 pPre/pPost 分别对齐到 0 与 kPostLocal，
        // 同时 GM 侧只从偏移 0 读取，避开非 32 字节对齐的源地址。
        const DataCopyExtParams lanePairCopy{2, kN * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams resCopy{1, kN * kN * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams gammaCopy{1, kWidth * sizeof(DT_PARAM), 0, 0, 0};
        if constexpr (kRawParam) {
            DataCopyPad(phiRaw_, phiGm_, phiCopy, {});
            DataCopyPad(biasRaw_, biasGm_, lanePairCopy, {});
            DataCopyPad(biasRaw_[kBiasRawRes], biasGm_[2U * kN], resCopy, {});
            if (hasGamma_) {
                DataCopyPad(gammaRaw_, gammaGm_, gammaCopy, {});
            }
        } else {
            DataCopyPad(phiFloat_, phiGm_, phiCopy, {});
            DataCopyPad(biasFloat_, biasGm_, lanePairCopy, {});
            DataCopyPad(biasFloat_[kResLocal], biasGm_[2U * kN], resCopy, {});
            if (hasGamma_) {
                DataCopyPad(gammaFloat_, gammaGm_, gammaCopy, {});
            }
        }
        SyncMte2ToVector();
        if constexpr (kRawParam) {
            Cast(phiFloat_, phiRaw_, RoundMode::CAST_NONE, kPhiLength);
            Cast(biasFloat_, biasRaw_, RoundMode::CAST_NONE, kN);
            Cast(biasFloat_[kPostLocal], biasRaw_[kBiasRawPost], RoundMode::CAST_NONE, kN);
            Cast(biasFloat_[kResLocal], biasRaw_[kBiasRawRes], RoundMode::CAST_NONE,
                 kN * kN);
            if (hasGamma_) {
                Cast(gammaFloat_, gammaRaw_, RoundMode::CAST_NONE, kWidth);
            }
        }
        Duplicate(reduceOut_, 0.0F, 8U);
        if constexpr (kN == 4U) {
            // n=4 时分组排布存在空洞，先清零避免读到未初始化数据。
            Duplicate(rowAcc_, 0.0F, kRowLength);
        }
        SetMaskNorm();
        // sigmoid 用 1/(1+exp(-z))，故把 pPre/pPost 两段的 bias 预先取反。
        SetVectorMask<float>(static_cast<int32_t>(kN));
        Muls<float, false>(biasFloat_, biasFloat_, -1.0F, 1U);
        Muls<float, false>(biasFloat_[kPostLocal], biasFloat_[kPostLocal], -1.0F, 1U);
    }

    __aicore__ inline void ProcessPosition(uint64_t position)
    {
        const BinaryRepeatParams contiguousBinary{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams contiguousUnary{1, 1, 8, 8};
        const DataCopyExtParams xCopy{1, kWidth * sizeof(DT_X), 0, 0, 0};
        if constexpr (kRawX) {
            DataCopyPad(xRaw_, xGm_[position * kWidth], xCopy, {});
        } else {
            DataCopyPad(xFloat_, xGm_[position * kWidth], xCopy, {});
        }
        SetVectorMask<float>(64);
        SyncMte2ToVector();
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
        if (hasGamma_) {
            Mul<float, false>(hInAcc_, xFloat_, gammaFloat_, MASK_PLACEHOLDER,
                              static_cast<uint8_t>(kWidthRepeats), contiguousBinary);
        } else {
            Muls<float, false>(hInAcc_, xFloat_, 1.0F, MASK_PLACEHOLDER,
                               static_cast<uint8_t>(kWidthRepeats), contiguousUnary);
        }

        const BinaryRepeatParams projectStrides{
            1, 1, 1, 8, 0, static_cast<uint8_t>(8U * kWidthRepeats)};
        Mul<float, false>(product_, hInAcc_, phiFloat_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRows), projectStrides);
        if constexpr (kWidthRepeats == 2U) {
            MulAddDst<float, float, false>(product_, hInAcc_[64U], phiFloat_[64U],
                                           MASK_PLACEHOLDER,
                                           static_cast<uint8_t>(kRows), projectStrides);
        }
        if constexpr (kN == 4U) {
            RepeatReduceSum<float, false>(rowAcc_, product_, static_cast<int32_t>(kN),
                                          MASK_PLACEHOLDER, 0, 1, 1, 8);
            RepeatReduceSum<float, false>(rowAcc_[kPostLocal], product_[kN * 64U],
                                          static_cast<int32_t>(kN), MASK_PLACEHOLDER, 0,
                                          1, 1, 8);
            RepeatReduceSum<float, false>(rowAcc_[kResLocal], product_[2U * kN * 64U],
                                          static_cast<int32_t>(kN * kN),
                                          MASK_PLACEHOLDER, 0, 1, 1, 8);
        } else {
            RepeatReduceSum<float, false>(rowAcc_, product_, static_cast<int32_t>(kRows),
                                          MASK_PLACEHOLDER, 0, 1, 1, 8);
        }

        SetVectorMask<float>(1);
        Muls<float, false>(reduceOut_, reduceOut_, invWidth_, 1U);
        Adds<float, false>(reduceOut_, reduceOut_, normEps_, 1U);
        Rsqrt<float, false>(reduceOut_, reduceOut_, MASK_PLACEHOLDER, 1U,
                            contiguousUnary);
        Brcb(product_, reduceOut_, 1U, {1, 8});

        SetVectorMask<float>(8);
        const BinaryRepeatParams broadcastStrides{1, 1, 1, 1, 1, 0};
        Mul<float, false>(rowAcc_, rowAcc_, product_, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRowLength / 8U), broadcastStrides);

        SetVectorMask<float>(static_cast<int32_t>(kN));
        Muls<float, false>(rowAcc_, rowAcc_, -alpha0_, 1U);
        Muls<float, false>(rowAcc_[kPostLocal], rowAcc_[kPostLocal], -alpha1_, 1U);
        Add<float, false>(rowAcc_, rowAcc_, biasFloat_, MASK_PLACEHOLDER, 1U,
                          contiguousBinary);
        Add<float, false>(rowAcc_[kPostLocal], rowAcc_[kPostLocal],
                          biasFloat_[kPostLocal], MASK_PLACEHOLDER, 1U,
                          contiguousBinary);
        SetVectorMask<float>(static_cast<int32_t>(kN * kN));
        Muls<float, false>(rowAcc_[kResLocal], rowAcc_[kResLocal], alpha2_, 1U);
        Add<float, false>(rowAcc_[kResLocal], rowAcc_[kResLocal], biasFloat_[kResLocal],
                          MASK_PLACEHOLDER, 1U, contiguousBinary);

        // 两段 sigmoid 相距一个 block，用 repeat=2 一次覆盖。
        SetVectorMask<float>(static_cast<int32_t>(kN));
        const UnaryRepeatParams pairStrides{1, 1, 1, 1};
        Exp<float, false>(sigmoidTmp_, rowAcc_, MASK_PLACEHOLDER, 2U, pairStrides);
        Adds<float, false>(sigmoidTmp_, sigmoidTmp_, 1.0F, 1U);
        Adds<float, false>(sigmoidTmp_[kPostLocal], sigmoidTmp_[kPostLocal], 1.0F, 1U);
        Reciprocal<float, false>(rowAcc_, sigmoidTmp_, MASK_PLACEHOLDER, 2U, pairStrides);
        Adds<float, false>(rowAcc_, rowAcc_, hcEps_, 1U);
        Muls<float, false>(rowAcc_[kPostLocal], rowAcc_[kPostLocal], 2.0F, 1U);

        Brcb(product_, rowAcc_, 1U, {1, 8});
        SetVectorMask<float>(8);
        Mul<float, false>(hInAcc_, xFloat_, product_, MASK_PLACEHOLDER, 2U,
                          broadcastStrides);
#pragma unroll
        for (uint32_t lane = 1U; lane < kN; ++lane) {
            MulAddDst<float, float, false>(hInAcc_, xFloat_[lane * kD],
                                           product_[lane * 8U], MASK_PLACEHOLDER, 2U,
                                           broadcastStrides);
        }

        if constexpr (kRawParam) {
            SetVectorMask<float>(static_cast<int32_t>(kN));
            Cast<DT_PARAM, float, false>(paramOutRaw_, rowAcc_[kPostLocal],
                                         RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1U,
                                         {1, 1, 4, 8});
            SetVectorMask<float>(static_cast<int32_t>(kN * kN));
            Cast<DT_PARAM, float, false>(paramOutRaw_[16U], rowAcc_[kResLocal],
                                         RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1U,
                                         {1, 1, 4, 8});
        }
        if constexpr (kRawX) {
            SetVectorMask<float>(static_cast<int32_t>(kD));
            Cast<DT_X, float, false>(xOutRaw_, hInAcc_, RoundMode::CAST_RINT,
                                     MASK_PLACEHOLDER, 1U, {1, 1, 4, 8});
        }

        SyncVectorToMte3();
        const DataCopyExtParams postCopy{1, kN * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams resStoreCopy{1, kN * kN * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams hInCopy{1, kD * sizeof(DT_X), 0, 0, 0};
        if constexpr (kRawParam) {
            DataCopyPad(hPostGm_[position * kN], paramOutRaw_, postCopy);
            DataCopyPad(hResGm_[position * kN * kN], paramOutRaw_[16U], resStoreCopy);
        } else {
            DataCopyPad(hPostGm_[position * kN], rowAcc_[kPostLocal], postCopy);
            DataCopyPad(hResGm_[position * kN * kN], rowAcc_[kResLocal], resStoreCopy);
        }
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
    LocalTensor<DT_PARAM> phiRaw_;
    LocalTensor<DT_PARAM> biasRaw_;
    LocalTensor<DT_PARAM> gammaRaw_;
    LocalTensor<DT_PARAM> paramOutRaw_;
    LocalTensor<float> xFloat_;
    LocalTensor<float> phiFloat_;
    LocalTensor<float> biasFloat_;
    LocalTensor<float> gammaFloat_;
    LocalTensor<float> product_;
    LocalTensor<float> hInAcc_;
    LocalTensor<float> reduceOut_;
    LocalTensor<float> sigmoidTmp_;
    LocalTensor<float> rowAcc_;

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

template <typename DT_X, typename DT_PARAM, int FAST_SMALL, int FAST_ONE_POSITION>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                   GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    if constexpr (FAST_SMALL != 0) {
        InitSocState();
        GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tilingData, tiling);
        if (tilingData.n == 4U) {
            KernelMhcPreSmallStatic<DT_X, DT_PARAM, 4U, FAST_ONE_POSITION != 0> op;
            op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
            op.Process();
        } else {
            KernelMhcPreSmallStatic<DT_X, DT_PARAM, 8U, FAST_ONE_POSITION != 0> op;
            op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
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
