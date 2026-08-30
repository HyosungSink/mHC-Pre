// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

template <class DT_X>
class KernelMhcPre {
public:
    __aicore__ inline KernelMhcPre() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_X>
 __global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tiling_data, tiling);
    KernelMhcPre<DT_X> op;
    op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tiling_data.length);
    op.Process();
}
