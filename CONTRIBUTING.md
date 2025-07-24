# 参与开发

使用 Python 3.10 以上版本。先创建虚拟环境，再安装 `pip install -e '.[dev]'`。
神经网络测试需要额外安装 CPU 版 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m build
python -m pytest -q
ruff check .
ruff format --check .
```

提交前按构建、测试、格式检查的顺序验证。提交信息使用 conventional commits；
版本默认仅增加 patch。新增行为应覆盖正常输入、边界和反向路径。
公共函数、类、字段和方法的签名由 `tests/contracts/public-api.json` 固定；
修改接口必须同时审查整个快照。

输入录音保存在本地。测试使用合成数据，不需要下载模型、语料或访问在线服务。
