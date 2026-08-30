# Kernel UT 接入位置

实现 Kernel 后在此接入 CANN `AscendcOpUt`、CPU 仿真或 Kernel 直调，覆盖：

- 完整 nD RMS 归约与 FP32 累加；
- gamma 有无及 hIn 必须读取原始 x；
- phi 转置投影、pre/post/res 切分和 hRes 行主序；
- sigmoid 极值、全零、大幅值、D 尾块与多核位置切分；
- FP16/BF16 输入输出转换及三个必选输出精度。

实际用例应复用 `tests/st/cases` 的分类样例和 `tests/common` 的共享数学 oracle。
