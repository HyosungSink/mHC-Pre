# mHcPre 单元测试

- `test_case_matrix.py` 检查 Coverage/Correctness 分类、输入契约、拒绝样例、
  流式数据配方和 manifest 中的分类标记。
- `test_reference.py` 检查 FP32 数学参考、输出排布、gamma、eps 和数值边界。
- `test_mock_cases.py` 检查固定 Mock 配置、fixture 大小与 Golden。
- `op_host/` 和 `op_kernel/` 是 CANN Host/Kernel 白盒 UT 的接入位置。

Python UT 通过 `python3 -m unittest discover -s tests/ut -t . -v` 运行。
设备侧正确性由 `tests/st/runner` 的 ACLNN Runner 检查。
