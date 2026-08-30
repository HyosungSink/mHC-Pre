# mHcPre 测试

测试按 Coverage、Correctness、Mock 三类组织。`common/` 保存共享契约和参考计算，
`st/cases/` 保存三类案例，`st/scripts/` 提供生成、审计和运行入口，`ut/` 验证测试资产。

| 类别 | 案例文件 | 数量 | 用途 |
|---|---|---:|---|
| Coverage | `st/cases/coverage.py` | 43 合法、54 拒绝 | `(B,S,n)` 全组合、D 边界、大形状和输入契约 |
| Correctness | `st/cases/correctness.py` | 9 合法 | 公式、排布、gamma、eps、零值与数值极端 |
| Mock | `st/cases/mock.json` | 2 | 混合精度正确性与本地设备诊断 |

`full` 包含 52 个正式合法样例和 54 个拒绝样例。Mock 使用独立的本地输入和运行入口，
不计入正式样例数量。Coverage 的 36 组 `(B,S,n)` 和拒绝输入由 Python 定义生成，
保持组合规则及单项错误构造集中可读；Mock 的固定参数保存在 JSON 中。

## 目录

~~~text
tests/
├── common/
│   ├── cases.py              # Tensor/调用契约、校验器与确定性输入配方
│   ├── reference.py          # FP32 数学参考
│   └── mock_cases.py         # Mock 配置加载、输入与 Golden
├── st/
│   ├── cases/
│   │   ├── coverage.py       # 输入范围及拒绝样例
│   │   ├── correctness.py    # 数值与排布样例
│   │   └── mock.json         # 两个固定本地 Mock
│   ├── runner/
│   │   └── mhc_pre_runner.cpp
│   └── scripts/
│   │   ├── audit_coverage.py # 完整案例目录审计
│   │   ├── audit_data.py     # 生成清单一致性检查
│   │   ├── gen_data.py       # 分类清单与覆盖报告
│   │   ├── gen_mock_data.py  # Mock 二进制 fixture
│   │   ├── verify_result.py  # 输出比较
│   │   └── run_tests.py      # 可移植测试流水线
├── ut/
│   ├── test_case_matrix.py
│   ├── test_reference.py
│   ├── test_mock_cases.py
│   ├── op_host/              # CANN Host UT 接入位置
│   └── op_kernel/            # CANN Kernel UT 接入位置
└── run_tests.py              # 原有命令的兼容入口
~~~

## 案例选择与自检

~~~bash
python3 -m tests.st.scripts.gen_data --suite coverage --list
python3 -m tests.st.scripts.gen_data --suite correctness --list
python3 -m tests.st.scripts.audit_coverage
python3 -m tests.st.scripts.run_tests --suite full
~~~

`--suite` 支持 `coverage`、`correctness`、`full`。`--list` 只列出案例，不写生成物。
运行入口默认使用临时目录；需要保留清单与报告时：

~~~bash
python3 -m tests.st.scripts.run_tests --suite full \
  --output /tmp/mhc_pre_st/data --coverage-report /tmp/mhc_pre_st/COVERAGE.md
python3 -m tests.st.scripts.gen_mock_data --output /tmp/mhc_pre_mock_data
~~~

测试流水线运行 Python UT、正式案例清单生成、一致性审计与比较器自检。
正式案例使用 `pattern + seed + element index` 流式输入配方；全维最大样例预计约
36.07 GiB，仅标记为 `shape_only`。Mock 入口物化输入和 Golden；
`st/runner/mhc_pre_runner.cpp` 可在 CANN 设备上检查三个必选输出并计时。

## ABI 范围

当前 Host 注册了题面混合精度布局和评测所需的全 fp16/全 fp32 布局。
`invRms/hMix/hPre` 三项成组可选输出尚未注册到 Host Schema。
请求它们的正式样例用于契约和参考计算覆盖；三个必选输出可由现有 ACLNN 接口运行。
