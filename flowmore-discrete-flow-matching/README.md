# FlowMore：离散流匹配名字生成（复现与实验）

FlowMore 是 Andrej Karpathy 的 makemore 的离散流匹配版本：读取 `names.txt` 中的名字列表，学习生成新的英文名字。

本仓库在原实现基础上完成复现实验，训练与生成结果的记录见 `FlowMore名字生成实验总结.md`。

## 文件说明

- `flowmore.py`：模型与训练主程序（DiT 主干 + 离散流匹配）
- `flowmore_reproduce.py`：复现实验脚本
- `names.txt`：训练用名字数据
- `assets/`：示例动画
- `outcomes/`：实验结果

## 怎么跑

```bash
pip install torch einops
python flowmore.py
```

## 来源说明

原实现基于离散流匹配论文框架，参考 Andrej Karpathy 的 makemore 改造而来。
