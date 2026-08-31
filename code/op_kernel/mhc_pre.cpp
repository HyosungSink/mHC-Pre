#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

template <class DT_X, class DT_PARAM>
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

        xRaw_ = LocalTensor<DT_X>(TPosition::VECCALC, kXRawAddress, kWidth);
        paramRaw_ = LocalTensor<DT_PARAM>(TPosition::VECCALC, kParamRawAddress, kParamLength);
        xFloat_ = LocalTensor<float>(TPosition::VECCALC, kXFloatAddress, kWidth);
        paramFloat_ = LocalTensor<float>(TPosition::VECCALC, kParamFloatAddress, kParamLength);
        product_ = LocalTensor<float>(TPosition::VECCALC, kProductAddress, kProductLength);
        hInAcc_ = LocalTensor<float>(TPosition::VECCALC, kHInAddress, kWidth);
        reduceOut_ = LocalTensor<float>(TPosition::VECCALC, kReduceAddress, 8U);
        scalarIn_ = LocalTensor<float>(TPosition::VECCALC, kScalarInAddress, 8U);
        scalarOut_ = LocalTensor<float>(TPosition::VECCALC, kScalarOutAddress, 8U);
        scalarAux_ = LocalTensor<float>(TPosition::VECCALC, kScalarAuxAddress, 8U);
        rowAcc_ = LocalTensor<float>(TPosition::VECCALC, kRowAddress, 32U);
        xOutRaw_ = LocalTensor<DT_X>(TPosition::VECCALC, kXOutAddress, 16U);
        paramOutRaw_ = LocalTensor<DT_PARAM>(TPosition::VECCALC, kParamOutAddress, 32U);
    }

    __aicore__ inline void Process()
    {
        LoadConstants();
        const uint32_t blockIndex = GetBlockIdx();
        const uint32_t blockCount = GetBlockNum();
        bool waitsForPreviousStore = false;
        for (uint64_t position = blockIndex; position < positionCount_; position += blockCount) {
            LoadX(position * kWidth);
            if (waitsForPreviousStore) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            SyncMte2ToVector();
            if constexpr (!IsSameType<DT_X, float>::value) {
                Cast(xFloat_, xRaw_, RoundMode::CAST_NONE, kWidth);
                PipeBarrier<PIPE_V>();
            }

            const float invRms = ProjectAndInvRms();
            float hPre0;
            float hPre1;
            float hPre2;
            float hPre3;
            Transform(invRms, hPre0, hPre1, hPre2, hPre3);
            ComputeHIn(hPre0, hPre1, hPre2, hPre3);
            StoreOutputs(position);

            waitsForPreviousStore = position + blockCount < positionCount_;
            if (waitsForPreviousStore) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    static constexpr uint32_t Align32(uint32_t value)
    {
        return (value + 31U) & ~31U;
    }

    static constexpr uint32_t kWidth = 64U;
    static constexpr uint32_t kRows = 24U;
    static constexpr uint32_t kProductLength = kWidth * kRows;
    static constexpr uint32_t kBiasOffset = kProductLength;
    static constexpr uint32_t kAlphaOffset = 1568U;
    static constexpr uint32_t kGammaOffset = 1584U;
    static constexpr uint32_t kParamLength = 1648U;

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
        Align32(kRowAddress + 32U * sizeof(float));
    static constexpr uint32_t kParamOutAddress =
        Align32(kXOutAddress + 16U * sizeof(DT_X));

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

    __aicore__ inline void LoadConstants()
    {
        const DataCopyExtParams phiCopy{1, kProductLength * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams biasCopy{1, kRows * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams alphaCopy{1, 3U * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams gammaCopy{1, kWidth * sizeof(DT_PARAM), 0, 0, 0};
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            DataCopyPad(paramFloat_, phiGm_, phiCopy, {});
            DataCopyPad(paramFloat_[kBiasOffset], biasGm_, biasCopy, {});
            DataCopyPad(paramFloat_[kAlphaOffset], alphaGm_, alphaCopy, {});
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
        SyncMte2ToVector();
        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            Cast(paramFloat_, paramRaw_, RoundMode::CAST_NONE, kParamLength);
            PipeBarrier<PIPE_V>();
        }

        if (hasGamma_) {
            const BinaryRepeatParams scaleStrides{1, 1, 1, 8, 0, 8};
            Mul(paramFloat_, paramFloat_[kGammaOffset], paramFloat_,
                static_cast<uint64_t>(kWidth), static_cast<uint8_t>(kRows), scaleStrides);
            PipeBarrier<PIPE_V>();
        }
        SyncVectorToScalar();
        alpha0_ = paramFloat_.GetValue(kAlphaOffset);
        alpha1_ = paramFloat_.GetValue(kAlphaOffset + 1U);
        alpha2_ = paramFloat_.GetValue(kAlphaOffset + 2U);
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

    __aicore__ inline float ProjectAndInvRms()
    {
        Mul(product_, xFloat_, xFloat_, kWidth);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduceOut_, product_, static_cast<int32_t>(kWidth), 1, 1, 1, 8);

        const BinaryRepeatParams projectionStrides{1, 1, 1, 8, 0, 8};
        Mul(product_, xFloat_, paramFloat_, static_cast<uint64_t>(kWidth),
            static_cast<uint8_t>(kRows), projectionStrides);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(rowAcc_, product_, static_cast<int32_t>(kWidth),
                       static_cast<int32_t>(kRows), 1, 1, 8);
        SyncVectorToScalar();
        const float meanWithEps = reduceOut_.GetValue(0) * invWidth_ + normEps_;

        Duplicate(scalarIn_, meanWithEps, 8U);
        PipeBarrier<PIPE_V>();
        Rsqrt(scalarOut_, scalarIn_, 8U);
        SyncVectorToScalar();
        return scalarOut_.GetValue(0);
    }

    __aicore__ inline void Transform(float invRms, float &hPre0, float &hPre1,
                                      float &hPre2, float &hPre3)
    {
#pragma unroll
        for (uint32_t index = 0; index < 4U; ++index) {
            scalarAux_.SetValue(index, invRms * alpha0_);
            scalarAux_.SetValue(4U + index, invRms * alpha1_);
        }
        SyncScalarToVector();
        Mul(rowAcc_, rowAcc_, scalarAux_, 8U);
        Muls(rowAcc_[8U], rowAcc_[8U], invRms * alpha2_, 16U);
        PipeBarrier<PIPE_V>();
        Add(rowAcc_, rowAcc_, paramFloat_[kBiasOffset], kRows);
        PipeBarrier<PIPE_V>();

        Muls(scalarOut_, rowAcc_, -1.0F, 8U);
        PipeBarrier<PIPE_V>();
        Exp(scalarOut_, scalarOut_, 8U);
        PipeBarrier<PIPE_V>();
        Adds(scalarOut_, scalarOut_, 1.0F, 8U);
        Duplicate(scalarAux_, 1.0F, 8U);
        PipeBarrier<PIPE_V>();
        Div(rowAcc_, scalarAux_, scalarOut_, 8U);
        PipeBarrier<PIPE_V>();
        Adds(rowAcc_, rowAcc_, hcEps_, 4U);
        SyncVectorToScalar();

#pragma unroll
        for (uint32_t index = 0; index < 4U; ++index) {
            scalarAux_.SetValue(index, 1.0F);
            scalarAux_.SetValue(4U + index, 2.0F);
        }
        SyncScalarToVector();
        Mul(rowAcc_, rowAcc_, scalarAux_, 8U);
        SyncVectorToScalar();

        hPre0 = rowAcc_.GetValue(0);
        hPre1 = rowAcc_.GetValue(1);
        hPre2 = rowAcc_.GetValue(2);
        hPre3 = rowAcc_.GetValue(3);
#pragma unroll
        for (uint32_t index = 0; index < 4U; ++index) {
            scalarIn_.SetValue(index, rowAcc_.GetValue(4U + index));
        }
    }

    __aicore__ inline void ComputeHIn(float hPre0, float hPre1, float hPre2, float hPre3)
    {
        SyncScalarToVector();
        Muls(hInAcc_, xFloat_, hPre0, 16U);
        Muls(product_, xFloat_[16U], hPre1, 16U);
        PipeBarrier<PIPE_V>();
        Add(hInAcc_, hInAcc_, product_, 16U);
        PipeBarrier<PIPE_V>();
        Muls(product_, xFloat_[32U], hPre2, 16U);
        PipeBarrier<PIPE_V>();
        Add(hInAcc_, hInAcc_, product_, 16U);
        PipeBarrier<PIPE_V>();
        Muls(product_, xFloat_[48U], hPre3, 16U);
        PipeBarrier<PIPE_V>();
        Add(hInAcc_, hInAcc_, product_, 16U);

        if constexpr (!IsSameType<DT_PARAM, float>::value) {
            Cast(paramOutRaw_, scalarIn_, RoundMode::CAST_RINT, 4U);
            Cast(paramOutRaw_[16U], rowAcc_[8U], RoundMode::CAST_RINT, 16U);
        }
        if constexpr (!IsSameType<DT_X, float>::value) {
            Cast(xOutRaw_, hInAcc_, RoundMode::CAST_RINT, 16U);
        }
    }

    __aicore__ inline void StoreOutputs(uint64_t position)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const DataCopyExtParams postCopy{1, 4U * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams resCopy{1, 16U * sizeof(DT_PARAM), 0, 0, 0};
        const DataCopyExtParams hInCopy{1, 16U * sizeof(DT_X), 0, 0, 0};
        if constexpr (IsSameType<DT_PARAM, float>::value) {
            DataCopyPad(hPostGm_[position * 4U], scalarIn_, postCopy);
            DataCopyPad(hResGm_[position * 16U], rowAcc_[8U], resCopy);
        } else {
            DataCopyPad(hPostGm_[position * 4U], paramOutRaw_, postCopy);
            DataCopyPad(hResGm_[position * 16U], paramOutRaw_[16U], resCopy);
        }
        if constexpr (IsSameType<DT_X, float>::value) {
            DataCopyPad(hInGm_[position * 16U], hInAcc_, hInCopy);
        } else {
            DataCopyPad(hInGm_[position * 16U], xOutRaw_, hInCopy);
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

template <typename DT_X, typename DT_PARAM, int FAST_SMALL>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                   GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    if constexpr (FAST_SMALL != 0) {
        InitSocState();
        GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tilingData, tiling);
        KernelMhcPreSmallStatic<DT_X, DT_PARAM> op;
        op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
        op.Process();
        PipeBarrier<PIPE_ALL>();
    } else {
        GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tilingData, tiling);
        KernelMhcPre<DT_X, DT_PARAM> op;
        op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
        op.Process();
    }
}
