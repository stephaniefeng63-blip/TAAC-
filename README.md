# 2025 腾讯广告算法大赛(TAAC) RTB 序列召回方案

基于统一用户-物品序列的 Transformer，面向 RTB 实时竞价广告创意召回。

**核心关键词**：统一序列建模 · 多模态融合 · 混合 InfoNCE 对比学习 · LogQ 校正 · FAISS ANN 召回

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg) ![Transformer](https://img.shields.io/badge/Transformer-SASRec-9c27b0.svg) ![FAISS](https://img.shields.io/badge/Retrieval-FAISS_HNSW-00897b.svg)

---

## 简介

本项目是 **2025 腾讯广告算法大赛 (TAAC)** 赛题的召回侧解决方案，面向 **RTB (Real-Time Bidding) 实时竞价广告** 的创意召回任务。

赛题给定用户的历史广告交互序列（曝光、点击等行为），要求为每个用户从海量广告创意中召回最相关的 Top-K。本仓库为召回阶段的模型实现。

**核心思路**：把用户特征和物品交互行为交错拼接到同一条序列里，用一个 Transformer 同时建模「用户是谁」和「用户在做什么」，再以对比学习把序列表征对齐到物品 embedding 空间，推理时通过 FAISS 做近似最近邻 (ANN) 召回 Top-K 候选创意。相比传统双塔召回，该方案把用户静态画像、行为序列、时间上下文、多模态创意特征统一在一个自回归主干里，用一个模型一次性产出 query 向量和 item 向量。

---

## 核心特性

### 模型主干
* **SASRec 风格 Transformer**：RMSNorm + 多头自注意力 + Conv1d 前馈网络，支持 Pre-Norm / Post-Norm。
* **Flash Attention**：优先调用 `F.scaled_dot_product_attention`，无可用时回退到手动 softmax 实现。
* **统一用户-物品序列**：用户特征 (token_type=2) 与物品交互 (token_type=1) 交错入序，共享同一套注意力参数。

### 多字段特征融合

| 特征类型 | 说明 |
| --- | --- |
| User Sparse | 用户侧离散特征 (103/104/105/109)，独立 embedding 表 |
| Item Sparse | 物品侧离散特征 (100/117/111/118/101/102 等 13 路) |
| Array 特征 | 变长离散序列特征，pooling 后融合 |
| 多模态创意 | 预训练 creative 向量 (81~86，维度 32~4096)，经线性投影对齐到隐藏维 |
| 时间连续特征 | sin / cos 周期编码 (小时/星期)、log 时间间隔、session 位置 |
| 时间桶 Embedding| 小时/星期/月/周/年/时段/是否周末，离散桶嵌入 |
| 贝叶斯平滑 CTR | 从原始曝光日志统计，按用户/物品计算平滑点击率作为连续特征 |

### 训练与推理
* **训练目标**：混合 InfoNCE 对比损失（1:1 配比的 Easy & Hard Negative）+ LogQ 流行度校正 + 冲突掩码。
* **训练工程**：AdamW + 余弦退火 + 混合精度 (AMP) 训练 + 梯度累积，支持 Checkpoint 自动续训与验证集评估。
* **召回推理**：序列末位 hidden state 作为 query；调用 FAISS HNSW 召回 Top-10；包含冷启动特征兜底处理。

---

## 架构总览

* **数据层 (`dataset.py`)**：读取行为序列、预训练多模态特征与贝叶斯 CTR 统计，进行特征工程，构建以 `token_type` 区分的统一序列。
* **模型层 (`model.py`)**：多字段特征融合后加入位置编码，输入 Transformer 主干提取最终序列表征。
* **训练流程 (`main.py`)**：计算混合 InfoNCE 损失，应用 LogQ 校正，通过余弦退火和 AMP 进行参数更新。
* **推理流程 (`infer`)**：导出用户 Query 向量和物品 Embedding 向量，使用 FAISS 执行 ANN 检索，输出最终 Top-10 创意。

---

## 项目结构

```text
TAAC--main/
├── main.py        # 训练入口：数据加载、训练循环、验证、checkpoint 保存
├── model.py       # BaselineModel：Transformer 主干 + 多字段特征融合
├── dataset.py     # 数据处理：特征工程、序列构造、CTR 统计
├── infer          # 推理脚本：导出向量 + FAISS ANN 召回 Top-10
└── README.md
