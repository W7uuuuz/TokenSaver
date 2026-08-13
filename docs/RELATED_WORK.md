# Related Work 调研（TokenSaver 论文定位）

> 2026-08-07 整理。目的：明确 TokenSaver 在大模型视频理解帧压缩研究中的位置，
> 支撑论文 Related Work 章节。文献以 2024-2026 为主，聚焦「视频帧/token 压缩」与
> 「文本/字幕引导选帧」两条线。

## 0. 领域地图

大模型视频理解（Video-LLM）面临的核心问题：视频展开为海量视觉 token，超出模型
上下文与算力预算。现有压缩手段分**三个层面**：

```
视频帧序列
   │
   ├─ ① 帧选择（编码前，选哪些帧进编码器）
   │     AKS / F2C / KFFocus / TokenSaver（本文）
   │
   ├─ ② token 剪枝（编码后，帧内 patch token 去冗余）
   │     FastV / SURGE / FLoC / VidCom2 / EVS
   │
   └─ ③ 空间/时域压缩（稀疏注意力、KV 缓存复用）
         VideoMamba 类、DyCoke
```

① 与 ② 互补：帧选择砍时间冗余，token 剪枝砍空间冗余。TokenSaver 属于 ①。

---

## 1. 帧选择（Frame / Keyframe Selection）

### 1.1 AKS — Adaptive Keyframe Sampling
- **出处**：Tang et al., CVPR 2025（arXiv:2502.21271）
- **方法**：训练无关、即插即用，插在 MLLM 视觉编码器之前。选帧目标 = **相关性**
  （帧与问题的匹配，用轻量 VL 模型如 BLIP/CLIP 打分）+ **覆盖度**（递归二分保证时间跨度）。
  在 Top 采样与均匀分箱之间自适应（ADA 算法）。
- **效果**：LLaVA-Video-7B 在 LongVideoBench 58.9→62.7，VideoMME 64.4→65.3。
- **与 TokenSaver 的关系**：同属「编码前帧选择」，但打分信号不同——AKS 用视觉
  **查询相关性**（需跑 BLIP/CLIP 编码每一帧），TokenSaver 用 **ASR 文本信息密度**
  （零视觉编码，纯文本预计算）。AKS 是 query-aware，TokenSaver 是 query-agnostic。

### 1.2 F2C — From Frames to Clips
- **出处**：Sun et al., 2025（arXiv:2510.02262）
- **方法**：论证帧级选择丢失时序动态，改为选**关键 clip**（短时连贯片段）+ 自适应
  分辨率，在固定 token 预算内平衡空间分辨率与片段长度。
- **效果**：Video-MME +8.1，LongVideoBench +5.6，MLVU +10.3（vs 均匀采样）。
- **与 TokenSaver 的关系**：F2C 提升维度是「时间动态/动作连续性」；TokenSaver
  关注「文本信息密度时刻」。两者可互补（clip 级 vs 帧级）。

### 1.3 KFFocus
- **出处**：Fudan University, CoRR 2025
- **方法**：从视频压缩原理出发，按**时间冗余**给帧分配不同凝聚比，结合时空建模。
- **与 TokenSaver 的关系**：KFFocus 用视觉帧间冗余，TokenSaver 用文本冗余（anti_redundancy
  维度）。都强调"去掉冗余、保留信息"。

### 1.4 其他 follow-ups
- **MaxInfo**（最大体积原则）、**MDP3**（马尔可夫决策 DPP）、**AdaRD-Key**（相关性-多样性）、
  **Q-Frame**（query-aware 多分辨率）、**LDDR**（线性 DPP 动态分辨率）、**FOCUS**。
- 共性：视觉特征打分 + 多样性约束。TokenSaver 是其中**唯一纯文本密度**、无训练、无 embedding 的方案。

---

## 2. Token 剪枝（帧内 patch token 去冗余）

### 2.1 FastV
- **出处**：Chen et al., 2024（arXiv:2403.06764）
- **方法**：发现深 Transformer 层中视觉 token 的注意力占比极低，在第 K 层后按注意力
  分数剪掉底部 R% 视觉 token。~45% FLOPs 节省，几乎无损，视频任务上甚至小幅提升。
- **与 TokenSaver 的关系**：FastV 在编码后、需访问注意力图；TokenSaver 在编码前、
  不需要模型参与。两者正交可叠加。

### 2.2 SURGE — Surprise-Guided Token Reduction
- **出处**：ICLR 2026
- **方法**：attention-agnostic（不依赖注意力图），用轻量常速预测器在 token 空间的
  **预测误差**作为"惊奇度"，剪掉低惊奇 token。Qwen2.5-VL 上压缩到 ~14-16% token，
  精度 ±1 点内，prefill FLOPs -86%，延迟 -79%。可选变体 SURGE⋆ 用 CLIP 聚焦 query 相关事件。
- **与 TokenSaver 的关系**：SURGE 的"惊奇度/预测误差"与 TokenSaver 的 **surprisal
  维度（权重 0.40）** 思想同源——都是"与预期偏离越多越值得保留"。但 SURGE 作用于
  token 空间（需先编码），TokenSaver 作用于**文本词空间**（无需编码）。

### 2.3 FLoC / VidCom2 / EVS
- **FLoC**：facility-location 子模函数选代表性+多样性 token（Qwen2.5-VL，支持到 7200 帧）。
- **VidCom2**（EMNLP 2025）：多尺度高斯相似度打分 + 动态帧预算，已集成 vLLM 支持 Qwen3-VL。
- **EVS**（vLLM 内置）：按与前帧时间差异剪 token，首帧全保留。

### 2.4 重要实证：随机剪枝是强基线
- Wen et al. (2025)：随机选帧/池化可匹敌注意力剪枝（注意力分数存在位置偏差与不稳定性）。
- 启示：**简单均匀/随机基线必须纳入对照**——TokenSaver 的实验设计一直以 Uniform 为
  基线对照，与此一致。

---

## 3. 字幕/ASR 引导选帧（与 TokenSaver 最同类）

### 3.1 双模态字幕引导（专利与早期系统）
- **US 11,386,665**：帧特征（HOG/SIFT/颜色直方图）+ 字幕文本特征（BoW/TF-IDF/词向量）
  → RNN/LSTM 打分 → DPP 选帧。**需训练**。
- **VidSynapse**：视频帧 embedding + ASR 文本 embedding，FAISS 索引 + 余弦聚类选代表帧
  （TVSum F1 59.5，SumMe 49.8）。**需跨模态 embedding**。
- **TriPSS**（arXiv:2506.05395）：感知（CIELAB）+ 结构（ResNet-50）+ 语义
  （Llama-3.2-Vision 生成 caption）三模态，HDBSCAN 聚类取 medoid 帧（TVSum F1 0.6104）。

### 3.2 字幕作为高密度信号（工程实践）
- **claude-video /watch、claude-real-video**：优先读原生字幕轨（免费、即时、高密度），
  无字幕才回退 Whisper；甚至纯字幕模式跳过帧。字幕 ~26.6k token vs 帧 ~9.8-22.8k token。
- 印证：**口播文本是视频中最廉价的语义信号**——这正是 TokenSaver 的立足点。

### 3.3 极压缩摘要
- **Dual Video Summarization**（IJCAI 2023）：压缩到不超过原视频 3% 的帧仍保持/提升
  摘要质量。
- **Multi-VidSum**（arXiv:2312.01575）：形式化为「关键帧-字幕对」联合生成任务，提供
  12,009 视频人工标注数据集。**其评估维度（关键帧匹配、METEOR、BLEURT）与 TokenSaver
  的 L1/L2 双轨评估思想一致**。

---

## 4. TokenSaver 差异化定位矩阵

| 维度 | AKS | F2C/KFFocus | FastV/SURGE/FLoC | 字幕引导类 | **TokenSaver** |
|---|---|---|---|---|---|
| 作用层级 | 编码前 | 编码前 | 编码后 | 编码前 | **编码前** |
| 打分信号 | CLIP/视觉 | 视觉冗余 | 注意力/惊奇度 | 跨模态 embedding | **ASR 文本密度** |
| 是否需视觉编码器 | 是（每帧） | 是 | 是 | 是（embedding） | **否（纯文本）** |
| 是否需训练 | 否 | 否 | 否 | 多数是 | **否** |
| Query-aware | 是 | 否 | 否（SURGE⋆ 是） | 否 | **否** |
| 额外算力成本 | 中（CLIP 编码） | 中 | 低-中（需先编码） | 高（embedding+聚类） | **近零** |

**核心创新主张**：
1. **首个纯 ASR 文本信息密度选帧**——5 维密度合成（surprisal 0.40 / 实体密度 0.31 /
   shannon 熵 0.12 / 反冗余 0.10 / TTR 0.07），零视觉编码成本，O(1) 预计算即可用。
2. **均匀基线兜底结构**——「均匀覆盖 + 关键帧增量」，ASR 完全退化时退化为均匀采样
   不塌缩（见 E-研究：ASR 鲁棒性）。
3. **L1（人工标注画面关键帧匹配）+ L2（内容理解保持率）双轨评估**——现有研究多只报
   理解基准（Video-MME 等），鲜有直接用人工标注关键帧做匹配评估。

---

## 5. 引用列表

1. Tang et al., *Adaptive Keyframe Sampling for Long Video Understanding*, CVPR 2025.
   https://arxiv.org/abs/2502.21271
2. Sun et al., *From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding*, 2025.
   https://arxiv.org/abs/2510.02262
3. Chen et al., *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration
   for Large Vision-Language Models*, 2024. https://arxiv.org/abs/2403.06764
4. *SURGE: Surprise-Guided Token Reduction for Efficient Video Understanding with VLMs*, ICLR 2026.
5. KFFocus, Fudan University, CoRR 2025.
6. VidCom2, EMNLP 2025（vLLM PR #47750）。
7. US Patent 11,386,665.
8. VidSynapse, EUDL 2025. https://eudl.eu/pdf/10.4108/eai.28-4-2025.2357767
9. *TriPSS: A Tri-Modal Keyframe Extraction Framework*, 2025. https://arxiv.org/abs/2506.05395
10. Song et al., *Dual Video Summarization*, IJCAI 2023. https://dl.acm.org/doi/abs/10.24963/ijcai.2023/94
11. *A Challenging Multimodal Video Summary: Keyframe-Caption Pairs*, 2023. https://arxiv.org/abs/2312.01575
12. Wen et al., random pruning strong baseline, 2025.
