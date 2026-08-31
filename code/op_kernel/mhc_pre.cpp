#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

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
        postLocalOffset_ = ((n_ + 7U) / 8U) * 8U;
        resLocalOffset_ = postLocalOffset_ + ((n_ + 7U) / 8U) * 8U;
        tileLength_ = tiling.tileLength;
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

        pipe_.InitBuffer(xRawBuf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(paramRawBuf_, tileLength_ * sizeof(DT_PARAM));
        pipe_.InitBuffer(xFloatBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(paramFloatBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(xGammaBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(productBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(hInAccBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(reduceTmpBuf_, 32);
        pipe_.InitBuffer(reduceOutBuf_, 32);
        pipe_.InitBuffer(scalarInBuf_, 32);
        pipe_.InitBuffer(scalarOutBuf_, 32);
        pipe_.InitBuffer(scalarAuxBuf_, 32);
        const uint32_t rowBufferLength = resLocalOffset_ + n_ * n_;
        const uint32_t rowBytes = ((rowBufferLength * sizeof(float) + 31U) / 32U) * 32U;
        pipe_.InitBuffer(rowAccBuf_, rowBytes);

        xRaw_ = xRawBuf_.Get<DT_X>();
        paramRaw_ = paramRawBuf_.Get<DT_PARAM>();
        xFloat_ = xFloatBuf_.Get<float>();
        paramFloat_ = paramFloatBuf_.Get<float>();
        xGamma_ = xGammaBuf_.Get<float>();
        product_ = productBuf_.Get<float>();
        hInAcc_ = hInAccBuf_.Get<float>();
        reduceTmp_ = reduceTmpBuf_.Get<float>();
        reduceOut_ = reduceOutBuf_.Get<float>();
        scalarIn_ = scalarInBuf_.Get<float>();
        scalarOut_ = scalarOutBuf_.Get<float>();
        scalarAux_ = scalarAuxBuf_.Get<float>();
        rowAcc_ = rowAccBuf_.Get<float>();
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
    TBuf<TPosition::VECIN> xRawBuf_;
    TBuf<TPosition::VECIN> paramRawBuf_;
    TBuf<TPosition::VECIN> xFloatBuf_;
    TBuf<TPosition::VECIN> paramFloatBuf_;
    TBuf<TPosition::VECIN> xGammaBuf_;
    TBuf<TPosition::VECIN> productBuf_;
    TBuf<TPosition::VECIN> hInAccBuf_;
    TBuf<TPosition::VECIN> reduceTmpBuf_;
    TBuf<TPosition::VECIN> reduceOutBuf_;
    TBuf<TPosition::VECIN> scalarInBuf_;
    TBuf<TPosition::VECIN> scalarOutBuf_;
    TBuf<TPosition::VECIN> scalarAuxBuf_;
    TBuf<TPosition::VECIN> rowAccBuf_;

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

template <typename DT_X, typename DT_PARAM>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                   GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tilingData, tiling);
    KernelMhcPre<DT_X, DT_PARAM> op;
    op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tilingData);
    op.Process();
}
