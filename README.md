# Kalman Filter: From 1D Scalar to 2D Matrix

一个大一学生学习卡尔曼滤波的笔记，记录了一些我当时没有理清楚的逻辑链条，python仿真也使得我们可以初步体会卡尔曼滤波的超参数调整

## 文件清单

```
kalman/
├── kf_1d_sim.py              # 一维标量 KF
├── kf_2d_sim.py              # 二维矩阵 KF（状态 = [位置, 速度]ᵀ）
├── kalman-filter-notes.md    # 详细学习笔记（理论 + 调参 + NumPy 踩坑）
└── README.md                 # 本文件
```

## 快速开始

```bash
# 1. 安装依赖（仅需 numpy + matplotlib）
pip install numpy matplotlib

# 2. 运行一维仿真
python kf_1d_sim.py

# 3. 运行二维仿真
python kf_2d_sim.py
```

两个脚本会自动弹出结果图，并将图片保存到脚本所在目录。

## 推荐学习路径

1. **先跑** `kf_1d_sim.py` — 所有矩阵退化为标量，观察 P 的锯齿和 K 的收敛过程
2. **再读** `kalman-filter-notes.md` 第 1-2 章 — 搞懂符号、五个公式、H 和 P 矩阵
3. **再跑** `kf_2d_sim.py` — 尝试修改 `q`（第 18 行）和 `R`（第 20 行），看滤波器性格变化
4. **对照片** `kalman-filter-notes.md` 第 5 章调参指南，理解 Q/R 比值如何影响结果
5. **避坑** 阅读第 6 章的 NumPy 常见陷阱（1D vs 2D 数组、`@` vs `*`、`np.outer` 等）

## 调参速查

| 你看到的 | 诊断 | 调法 |
|----------|------|------|
| KF 曲线紧跟测量值，抖动大 | Q 太大（信模型不够）| `q` ↓ (第 17-18 行的 `* 1` 改小) |
| KF 曲线平滑但总慢半拍 | Q 太小（信模型太多）| `q` ↑ |
| 前几步震荡很大才收敛 | P_init 太小（初始太自信）| `P_Init_matrix` × 10 |
| 前几步收敛太慢 | P_init 太大 | `P_Init_matrix` ÷ 10 |

`q` 和 `R` 的定义见 `kf_2d_sim.py` 第 17-20 行。详细调参原理见笔记第 5 章。

## 依赖

- Python 3.6+
- NumPy ≥ 1.18
- Matplotlib ≥ 3.0

