# 【A组困难题】mHcPre 算子

---

## 1. 算子说明

Mhc Pre 是 mHC（流形约束超连接，Manifold-Constrained Hyper-Connections）架构的**前处理算子**：对 mHC 层的输入 `x` 做 RmsNorm 归一化与参数矩阵投影，计算出 hidden 层的 `Hres`/`Hpost` 投影矩阵（供后续 MhcSinkhorn/MhcPost 算子使用）以及 Attention/MLP 层的输入 `hIn`。它将输入 x 的最后两维 (n, D) 展平后做 RmsNorm、与参数矩阵 φ 相乘、经 sigmoid 激活与缩放，最终加权求和得到 hIn。

### 背景术语（题目内自解释）

- **mHC 架构**：Manifold-Constrained Hyper-Connections，一种将残差连接与隐层投影融合的网络结构，通过 `Hres`/`Hpost` 矩阵对信号做投影变换，稳定深度网络传播。本算子是其前处理，输出供 MhcSinkhorn（双随机化）与 MhcPost（后处理）使用。
- **RmsNorm**：Root Mean Square Normalization，按均方根做归一化：`invRms = 1/√(mean(x²)+normEps)`，再用 `invRms` 缩放。
- **xFlat**：将 x 的最后两维 `n` 与 `D` 视作长度为 `nD` 的向量（即 reshape(nD)），便于与参数矩阵做矩阵乘。
- **sigmoid**：`σ(z) = 1/(1+e^(-z))` 激活函数。
- **alpha 模式**：`alpha.shape=[3]`，输出含残差 `hRes`（phi 为 `n²+2n` 行）。
- **维度符号**：`B`=Batch Size，`S`=Sequence Length，`n`=超连接矩阵维度（支持 4/6/8），`D`=隐藏层维度（1~16384，需 16 对齐）。
- **数据排布**：本算子输入 x 为 4 维 `(B, S, n, D)`（ND 格式，非 TND）。

### 计算公式

`xFlat` 表示将 x 最后两维展平为长度 `nD` 的向量，`gammaFlat` 同理将 gamma 展平为 `nD` 向量，`@` 为矩阵乘，`⊙` 为逐元素乘：

```
invRms = (mean(xFlat²) + normEps)^(-1/2)
xGamma = xFlat ⊙ gammaFlat      (gamma ≠ null);  否则 xGamma = xFlat
hMix   = xGamma @ φ^T
w      = hMix ⊙ invRms
[pPre, pPost, pRes] = split(w, [n, n, n²])     # alpha.shape=[3]
hPre   = σ(pPre ⊙ alpha0 + bias0) + hcEps
hPost  = 2·σ(pPost ⊙ alpha1 + bias1)
hRes   = pRes ⊙ alpha2 + bias2
hIn_d  = Σ_{i=0}^{n-1} hPre_i · x_{i,d}
```

其中 `alpha0/alpha1/alpha2` 为 alpha 的三个分量，`bias0/bias1/bias2` 为 bias 按前 n、中 n、后 n² 切分。hIn 公式中 `i` 遍历 x 的 n 维（倒数第二维），`d` 遍历 x 的 D 维（最后一维），`hPre_i` 为 hPre 在 n 维的第 i 个分量。

---

## 2.  输入输出说明

### 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| x | 必选 | 张量 | (B, S, n, D) | bfloat16/float16 | mHC 层输入数据，公式中的 x。不能为空 |
| phi | 必选 | 张量 | (n²+2n, nD) | float32 | mHC 参数矩阵 φ |
| alpha | 必选 | 张量 | (3,) | float32 | mHC 缩放参数 |
| bias | 必选 | 张量 | (n²+2n,) | float32 | mHC bias 参数，与 phi 行数对应 |
| gamma | 可选 | 张量 | (n, D) | float32 | RmsNorm 缩放因子。推理场景可传空（此时 xGamma=xFlat） |
| normEps | 可选 | 属性 | 标量 | double | RmsNorm 防除零参数 normEps。建议值 1e-6 |
| hcEps | 可选 | 属性 | 标量 | double | hPre sigmoid 后的 eps 参数 hcEps。建议值 1e-6 |

### 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| hIn | 必选 | (B, S, D) | bfloat16/float16 | Attention/MLP 层输入 hIn |
| hPost | 必选 | (B, S, n) | float32 | mHC 的 hPost 变换矩阵 |
| hRes | 必选 | (B, S, n, n) | float32 | mHC 的 hRes 变换矩阵（未做 sinkhorn） |
| invRms | 可选 | (B, S) | float32 | RmsNorm 的 1/r。与 hMix/hPre 互存（同时输出或全不输出） |
| hMix | 可选 | (B, S, n²+2n) | float32 | xGamma 与 φ^T 矩阵乘结果。与 invRms/hPre 互存 |
| hPre | 可选 | (B, S, n) | float32 | sigmoid 后的 hPre 矩阵。与 invRms/hMix 互存 |

> 说明：可选输出 `invRms`、`hMix`、`hPre` 为**互存关系**，需同时输出或全部不输出（不支持仅返回其中部分），条件为三者指针均非空。

### 形状约束

- 批大小 `B` 仅支持 **1、2、4**。
- 序列长度 `S` 仅支持 **4k、8k、16k、32k**（即 4096、8192、16384、32768）。
- 矩阵维度 `n` 仅支持 **4、6、8**。
- 隐藏维度 `D` 取值 1~16384，需满足 **16 对齐**。
- phi 形状 `(n²+2n, nD)`、bias 形状 `(n²+2n,)`、hMix 形状 `(B,S,n²+2n)`、hRes 必输出 `(B,S,n,n)`。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。
- 默认确定性实现。

### 输出形状计算公式

- `hIn`：x 去掉 n 维，即 `(B, S, D)`。
- `hPost`：`(B, S, n)`。
- `hRes`：`(B, S, n, n)`。
- `invRms`：`(B, S)`；`hMix`：`(B, S, n²+2n)`；`hPre`：`(B, S, n)`。

---

## 3. 算子逻辑说明

### 3.1 RmsNorm 归一化与 xGamma

1. 将 x 最后两维 (n, D) 展平为长度 nD 的向量 `xFlat`（形状 (B,S,nD)）。
2. 计算 `invRms = 1/√(mean(xFlat², 沿 nD 维) + normEps)`（形状 (B,S,1)）。
3. 若 gamma 不为空，`xGamma = xFlat ⊙ gammaFlat`（gamma 展平为 nD）；否则 `xGamma = xFlat`。

### 3.2 投影 hMix 与归一化 w

1. `hMix = xGamma @ φ^T`（xGamma 形状 (B,S,nD)，φ^T 形状 (nD, n²+2n)，结果 (B,S,n²+2n)）。
2. `w = hMix ⊙ invRms`（用 invRms 对 hMix 做 RmsNorm 缩放）。

### 3.3 split 与 hPre/hPost/hRes

1. 将 w 沿最后一维切分为 `[pPre(n), pPost(n), pRes(n²)]`，其中 pRes reshape 为 (B,S,n,n)。
2. `hPre = σ(pPre ⊙ alpha0 + bias0) + hcEps`（bias0 = bias 前 n 个）。
3. `hPost = 2·σ(pPost ⊙ alpha1 + bias1)`（bias1 = bias 中间 n 个）。
4. `hRes = pRes ⊙ alpha2 + bias2`（bias2 = bias 后 n² 个，reshape 为 (n,n)）。

### 3.4 hIn 加权求和

`hIn_d = Σ_{i=0}^{n-1} hPre_i · x_{i,d}`：用 hPre 对 x 的 n 维做加权求和，得到每个 d 位置的输出，形状 (B,S,D)。

### 3.5 可选中间输出

当 invRms/hMix/hPre 三者指针均非空时，同时输出这三组中间结果（供反向梯度计算使用）；否则均不输出。

---

## 4. 任务要求

- **精度保障**：针对不同 `n`（4/6/8）、不同 `D` 与 gamma 有无，设计算子逻辑，保证 RmsNorm、投影、sigmoid、加权求和精度正确。
- **性能优化**：优化 xFlat 展平、`xGamma@φ^T` 矩阵乘与 hIn 加权求和的数据搬运与并行度，算子性能更优。
- **切分最优**：探索 B、S、nD 维度在多 batch/长序列场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 5.  功能示例

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mhc_pre(x, phi, alpha, bias, gamma=None, norm_eps=1e-6, hc_eps=1e-6):
    # x:(B,S,n,D)  phi:(n^2+2n, nD)  alpha:(3)  bias:(n^2+2n)  gamma:(n,D)或None
    # 输出 hIn:(B,S,D)  hPost:(B,S,n)  hRes:(B,S,n,n)  (+可选 invRms/hMix/hPre)
    xf = x.astype(np.float32)
    B, S, n, D = xf.shape
    xflat = xf.reshape(B, S, n * D)                                    # 最后两维展平为 nD
    inv_rms = 1.0 / np.sqrt(np.mean(xflat ** 2, axis=-1, keepdims=True) + norm_eps)  # (B,S,1)
    xgamma = xflat * gamma.reshape(n * D) if gamma is not None else xflat
    h_mix = xgamma @ phi.T                                             # 投影 (B,S, n^2+2n)
    w = h_mix * inv_rms                                                # RmsNorm 缩放
    p_pre = w[..., :n]; p_post = w[..., n:2 * n]
    p_res = w[..., 2 * n:].reshape(B, S, n, n)
    h_pre = sigmoid(p_pre * alpha[0] + bias[:n]) + hc_eps             # (B,S,n)
    h_post = 2 * sigmoid(p_post * alpha[1] + bias[n:2 * n])           # (B,S,n)
    h_res = p_res * alpha[2] + bias[2 * n:].reshape(n, n)             # (B,S,n,n)
    h_in = np.einsum('bsn,bsnd->bsd', h_pre, xf)                      # hPre 加权 x 求和 (B,S,D)
    return h_in.astype(np.float16), h_post, h_res, inv_rms, h_mix, h_pre

np.set_printoptions(precision=4, suppress=True)

# 示例：alpha=[3]（有 hRes 残差），gamma 非空（B=1, S=1, n=4, D=16）
# 为可读性使用较小 D=16，实际算子支持 D 取值 1~16384（16 对齐），典型值如 2560/4096
# x/phi/gamma 全 1，alpha=[1,1,1]，bias 全 0
B, S, n, D = 1, 1, 4, 16
x     = np.ones((B, S, n, D), dtype=np.float16)
phi   = np.ones((n * n + 2 * n, n * D), dtype=np.float32)   # (24, 64)
alpha = np.array([1, 1, 1], dtype=np.float32)
bias  = np.zeros(n * n + 2 * n, dtype=np.float32)
gamma = np.ones((n, D), dtype=np.float32)
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_pre(x, phi, alpha, bias, gamma)
# 输出形状：hIn=(1,1,16) hPost=(1,1,4) hRes=(1,1,4,4) invRms=(1,1) hMix=(1,1,24) hPre=(1,1,4)
# 全 1 输入下：invRms≈1、hMix=64、w=64、hPre≈1、hPost=2、hRes=64、hIn=4（hPre 对 n=4 个 1 求和）
print('hIn 首值(期望4):', float(h_in[0, 0, 0]))
print('hPost 首值(期望2):', float(h_post[0, 0, 0]))
print('hRes 首值(期望64):', float(h_res[0, 0, 0, 0]))
```

