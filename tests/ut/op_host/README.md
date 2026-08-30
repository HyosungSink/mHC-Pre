# Host UT 接入位置

实现 Host 后在此验证：

- B/S/n/D 推导与支持集合；
- phi、alpha、bias、gamma 的关联 Shape；
- 三个必选输出的 Shape/DType 推导；
- gamma 有无、tile 尾块、workspace、block dim 与 tiling key/data；
- 非法 rank、Shape、DType、ND/连续性和 ABI 不支持组合的拒绝行为。

优先采用 CANN `op_test_frame` 生成的 tiling UT 或项目可编译的 GTest。
