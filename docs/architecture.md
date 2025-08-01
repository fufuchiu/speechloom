# 架构与边界

```text
waveform ----> strided Conv1d ----> audio Transformer ----> encoder memory
                                                               |
BOS + text + audio codes ----> causal Transformer decoder <-----+
                                       |
                                shared vocabulary logits
                                       |
                      text / audio weighted cross entropy
```

端到端指损失梯度可以从文本和音频 token 输出传播到波形编码器。
音频前端步长为 16；输入填充在卷积前清零，编码器和解码器分别应用有效长度掩码。
解码使用严格上三角因果掩码。生成复用音频 memory，默认采用确定性贪心策略；
文本和音频模式各自限制可输出的 token 范围。

当前模型是 CPU 上可以验证的紧凑原型。mu-law 是逐采样量化演示，残差码本实现
用于低维特征实验；它们不是高保真神经语音编解码器。实际语音大模型训练还需要
合适的预训练底座、神经音频 tokenizer、语料、硬件与大规模评测。

流工具管理音频字节、UTF-8、背压与取消；模型音频编码本身处理整段输入，
不声称已经实现增量 KV cache 或实时全双工语音系统。

检查点只包含配置、版本、训练步数和模型参数；不包含优化器状态。
加载适用于推理和重新初始化优化器后的继续训练，不保证位级复现中断前的更新。
