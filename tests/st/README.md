# mHcPre 系统测试

`cases/` 分别保存 Coverage、Correctness、Mock。Coverage 含 43 个合法输入范围样例和
54 个契约拒绝样例；Correctness 含 9 个公式、数值极端和排布样例；Mock 含 2 个固定的
本地小尺寸设备诊断样例。

`common/cases.py` 提供共享契约和输入配方，`common/reference.py` 提供 FP32 Golden，
`common/mock_cases.py` 管理 Mock 输入与 Golden。`scripts/audit_coverage.py` 审计完整
正式案例目录；`scripts/gen_data.py` 按 suite 生成带分类标记的 manifest 和覆盖报告；
`scripts/audit_data.py` 核对生成清单；`scripts/verify_result.py` 比对输出。

~~~bash
python3 -m tests.st.scripts.gen_data --suite coverage --list
python3 -m tests.st.scripts.gen_data --suite correctness --list
python3 -m tests.st.scripts.run_tests --suite full
python3 -m tests.st.scripts.gen_mock_data --output /tmp/mhc_pre_mock_data
~~~

`runner/mhc_pre_runner.cpp` 可在设备上调用 ACLNN、检查 Mock 的三个必选输出并计时。
请求 `invRms/hMix/hPre` 的正式样例目前只做
契约覆盖，因为三项可选输出尚未注册到 Host Schema。
