# 模块索引

| 模块 | 职责 |
|---|---|
| `audio` | PCM16、Base64、波形批次与交叉淡化 |
| `codec` | mu-law 基线、k-means 码本与残差向量量化 |
| `tokens` | UTF-8 文本和多码本音频的互斥 token 空间 |
| `batching` | teacher forcing、因果掩码、填充掩码和 token 预算 |
| `sampling` | 温度、top-k、top-p、重复惩罚和模态限制 |
| `conversation` | 有类型的多轮对话清单和监督样本 |
| `streaming` | 有界队列、取消、PCM 与 UTF-8 增量解码 |
| `metrics` | 首 token 延迟、RTF、间隔分位数和 bootstrap |
| `model` | 可微音频编码、因果 Transformer、联合损失与检查点 |

每个公共入口的参数签名、默认值和数据类字段都由 `tests/contracts/public-api.json` 固定。
输入校验在计算前完成；无效维度、非有限数字和越界 ID 抛出 `ValueError`。
