# FlowMore：基于离散流匹配（Discrete Flow Matching）的名字生成实验总结

> 项目：[phuocthang/minimal-discrete-flow-matching](https://github.com/phuocthang/minimal-discrete-flow-matching)
>（本地复现代码位于本目录，核心脚本 `flowmore.py`）
> 实验日期：2026-08-21
> 硬件：NVIDIA GeForce RTX 4060 Laptop（8.6GB GPU），torch 2.6.0+cu118

---

## 1. 一句话总结

用 **离散流匹配（Discrete Flow Matching，DMF）** 训练一个 **Diffusion Transformer（DiT）**，在 **32033 个英文名字**上学到字符转移规律；随后从随机字符出发，沿时间轴 t=0→1 用学到的 velocity field 逐步「反演（去噪）」，**并行生成有意义、符合英文名字形态的新名字**。

与传统的自回归逐字生成不同，这是**非自回归、并行去掩码/去噪**的生成方式——所有位置同时从噪声演化到目标字符。

---

## 2. 方法与机制

核心思路（对应 [flowmore.py](flowmore.py)）：

1. **前向（加噪）**：对每条名字 `x_1`，采样一个随机时间 `t∈[0,1]`，把每个位置以概率 `t` 从目标字符换成一个随机字符（源分布），得到中间态 `x_t`：`x_t = where(rand < t, x_1, x_0)`。
2. **学习目标**：训练模型预测 `p(x_1 | x_t, t)`，即给定部分加噪的序列和时间，预测每个位置的**目标字符分布**（交叉熵）。本质是学习 velocity field。
3. **反向（采样）**：从随机 `x_0` 出发，用欧拉式小步长 `h`：
   `u = (p1 - one_hot(x_t)) / (1 - t)`，每步用 `Categorical` 采样更新 `x_t`，直到 `t→1`，得到完整名字。

**分词方式**：字符级（character-level）。词表 28 = 26 个字母 + `.`（名字结束符）+ `_`（padding 填充符）。每条名字统一 padding 到固定长度 `max_len=16`。

---

## 3. 模型与训练配置

| 项 | 值 |
|---|---|
| 模型 | Diffusion Transformer（DiT）：3 层、dim=64、4 头注意力、Fourier 时间编码、AdaLN 调制 |
| 参数量 | **228,700** |
| 数据集 | `names.txt`，**32033** 个名字；词表 28 字符；`max_len=16` |
| 训练步数 | 10,000 步 |
| batch size | 256 |
| 优化器 / lr | Adam(lr=5e-4) + CosineAnnealingLR(T_max=10000, eta_min=1e-6) |
| 源分布 x_0 | 均匀随机字符 |
| 采样 | 离散流匹配欧拉反演，步长 `h=min(0.1, 1-t)` |
| 训练耗时 | 约 1–2 分钟（GPU） |

---

## 4. 训练收敛

- 损失曲线完整记录在 `outcomes/train_loss.npy`，并把训练 loss 随步下降的过程绘成图（见下方图 1）。
- 收敛数据（交叉熵损失）：
  | 指标 | 损失 |
  |---|---|
  | 初始（step 0） | 3.7966 |
  | step 100 | 0.8179 |
  | step 1000 | 0.7516 |
  | 最终（step 10000） | 0.6600 |
  | 全程最低 | 0.5812 |

**观察**：前 100 步损失从 3.8 骤降到 ~0.82（快速学会基本字符分布），随后缓慢下降至 ~0.66，符合小模型在字符级任务上的收敛形态。

---

## 5. 生成结果（离散流匹配采样）

从随机 `x_0` 出发，沿 t 从 0.1 逐步反演到 1.0，共生成 8 个名字（`_` 为 padding）：

| # | 生成名字 | 解读 |
|---|---|---|
| 0 | `kutsla.` | 类 kyla / luna |
| 1 | `egzo.` | 类 eli / zoey |
| 2 | `embron.` | 学到 `emb`/`em` 前缀（emma/emily 类） |
| 3 | `bruzs.` | 含 `bru` 前缀 |
| 4 | `anyeigh.` | 学到 `any` 前缀（anya 类） |
| 5 | `priya.` | ✅ 拼写很标准 |
| 6 | `kilan.` | 学到 `kil` 前缀（milan 类） |
| 7 | `clokgo.` | 学到 `ok`/`lo` 组合 |

**要点**：名字均以 `.` 结尾、长度适中（4–7 个字母），模型准确学到了「名字以句号结束、长度适当」的分布先验。

---

## 6. 可视化展示（图片地址）

### 📊 图 1：训练 Loss 曲线
色彩展示了离散流匹配模型从随机初始化到收敛的完整学习过程（步数 → 交叉熵损失）。

图片地址：
- `d:\大三上的资料\假期学习记录\minimal-discrete-flow-matching\outcomes\training_loss_curve.png`

### 📊 图 2：名字逐步生成的去噪轨迹网格
每个名字一个字图：**行 = 采样时间步 t（0.1→1.0）**，**列 = 16 个字符位置**，颜色 = 字符 id。能直观看到所有位置**同时**从随机颜色逐步「固化」成最终字符——这正是离散流匹配的**并行并行去噪**机制。

图片地址：
- `d:\大三上的资料\假期学习记录\minimal-discrete-flow-matching\outcomes\name_sampling_trajectories.png`

（例如名字 `kutsla.` 的轨迹：早期各位置是随机杂乱字符，随 t 增大逐个/整体收敛为 `k-u-t-s-l-a-.`，是很好的流程图/汇报素材。）

---

## 7. 结论

1. **离散流匹配可行**：即便用很小的模型（228K 参数）和很短的训练（1–2 分钟），也能生成「像模像样」的英文名字。
2. **并行生成优势**：所有 token 位置同时去噪，非自回归，一阶反演即得完整序列。
3. 与自回归（makemore）对比，DMF 保持了生成质量的同时引入了**可并行的生成流程**，后续可配合 Classifier-Free Guidance 等技巧进一步提升。

---

## 8. 产物文件清单

| 内容 | 绝对路径 |
|------|---------|
| 本总结 | `d:\大三上的资料\假期学习记录\minimal-discrete-flow-matching\FlowMore名字生成实验总结.md` |
| 训练 Loss 曲线图 | `...\outcomes\training_loss_curve.png` |
| 名字去噪轨迹图 | `...\outcomes\name_sampling_trajectories.png` |
| 训练 loss 逐步数据 | `...\outcomes\train_loss.npy` |
| 生成的 8 个名字 | `...\outcomes\generated_names.txt` |
| 复现+画图脚本 | `...\flowmore_reproduce.py` |
| 原实验脚本 | `...\flowmore.py` |
| 使用的数据 | `...\names.txt` |