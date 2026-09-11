# speechloom

端到端语音模型研究工具：音频编码、因果解码、联合训练与流式协议。

Chen Yuxuan

> 历史说明：本仓库由经过验证的补丁序列重建；2025 年至 2026 年 8 月的 Git 时间戳用于展示项目演进，不代表代码实际开发日期。

[![CI](https://github.com/fufuchiu/speechloom/actions/workflows/ci.yml/badge.svg)](https://github.com/fufuchiu/speechloom/actions/workflows/ci.yml)

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

```bash
python -m speechloom tokenize '大学' --audio-codes 0,128,255
python examples/codec_roundtrip.py
python examples/stream_text.py
```

文本采用无损 UTF-8 byte token。音频码本分配独立 ID 区间，避免模态碰撞。
训练时输出序列为 `BOS + text + separator + audio_codes + EOS`。

## 模型实验

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/train_tiny.py
```

提供可以训练的紧凑研究原型和完整数据链路。示例使用合成输入，验证损失下降、
梯度和掩码正确性；不包含经过大规模训练的权重，也没有真实语料成绩声明。

## 文档

- [架构与边界](docs/architecture.md)
- [模块索引](docs/api-reference.md)
- [复现实验与时间线说明](docs/reproducibility.md)
- [测试策略](docs/testing.md)
- [参与开发](CONTRIBUTING.md)

## 验证

```bash
python -m build
python -m pytest -q
ruff check .
ruff format --check .
```

MIT License · Chen Yuxuan
