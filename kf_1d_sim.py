"""
一维卡尔曼滤波仿真 — 跟踪一个匀速运动物体的位置
====================================================
用最简单的标量 KF（所有矩阵退化成单个数字），


运行方式：在终端输入  python kf_1d_sim.py
"""

import os
import numpy as np                    # 相当于 C 的 #include <math.h>，提供数组和数学运算
import matplotlib.pyplot as plt       # 相当于画图库，用来出图

# ============================================================
# 1. 仿真参数设定（相当于 C 的 #define 或 const）
# ============================================================
DT = 1.0              # 时间步长 Δt（秒）
TOTAL_STEPS = 50      # 总步数（相当于 for 循环跑 50 次）

TRUE_VELOCITY = 2.0   # 真实速度（物体以恒定速度运动）
TRUE_INIT_POS = 0.0   # 真实初始位置

Q = 0.1              # 过程噪声方差（模型的不确定性，我们设定的）
R = 1               # 观测噪声方差（传感器噪声，看传感器说明书）

# ============================================================
# 2. 分配数组（相当于 C 的 malloc）
# ============================================================
# 在 Python 里 np.zeros(N) 创建一个长度为 N 的全零数组，类似:
#   double *arr = (double*)calloc(N, sizeof(double));

true_pos  = np.zeros(TOTAL_STEPS)   # 真实位置（上帝视角，算法不可见）
measured  = np.zeros(TOTAL_STEPS)   # 传感器读数（带噪声的观测）
x_prior   = np.zeros(TOTAL_STEPS)   # 先验估计 x̂⁻（纯靠模型猜）
x_post    = np.zeros(TOTAL_STEPS)   # 后验估计 x̂（融合观测之后）
P_prior   = np.zeros(TOTAL_STEPS)   # 先验协方差 P⁻（猜完之后的不确定度）
P_post    = np.zeros(TOTAL_STEPS)   # 后验协方差 P（修正之后的不确定度）
K_values  = np.zeros(TOTAL_STEPS)   # 卡尔曼增益 K（每步算出来的权重）

# ============================================================
# 3. 生成"真实轨迹"和"传感器读数"（模拟物理世界）
# ============================================================
# 这相当于你在 RoboMaster 里跑了一遍真实的机器人，然后拿传感器记录数据

np.random.seed(42)  # 固定随机种子，每次运行结果一样（调试用）

for k in range(TOTAL_STEPS):        # for (k = 0; k < TOTAL_STEPS; k++)
    # 真实位置 = 初始位置 + 速度 × 时间
    true_pos[k] = TRUE_INIT_POS + TRUE_VELOCITY * (k * DT)

    # 传感器读数 = 真实位置 + 噪声（模拟传感器不准）
    # np.random.randn() 返回一个服从 N(0,1) 的随机数
    # sqrt(R) × N(0,1) = N(0, R)
    noise = np.sqrt(R) * np.random.randn()
    measured[k] = true_pos[k] + noise

# 打印前几行数据看看
print("k    true_pos    measured      noise")
print("-" * 45)
for k in range(5):
    print(f"{k}    {true_pos[k]:8.3f}    {measured[k]:8.3f}    {measured[k]-true_pos[k]:8.3f}")

# ============================================================
# 4. 卡尔曼滤波主循环
# ============================================================
# 这就是 KF 的灵魂——五个公式在循环里跑

# 初始条件：我们不知道物体在哪，所以先随便猜一个，给很大的 P
# 相当于 C:  double x_post_prev = 0.0, P_post_prev = 1000.0;

x_post_prev = 0.0       # 初始估计：猜在 0 位置
P_post_prev = 1000.0    # 初始不确定度：非常大（≈ 完全不知道在哪）

# 一维情况下 F = 1（位置不变，因为我们模型假设匀速但没把速度放进状态）
# H = 1（传感器直接测位置，无转换）
F = 1.0  # 状态转移系数
H = 1.0  # 观测系数

print("\n" + "=" * 70)
print("KF Main Loop")



for k in range(TOTAL_STEPS):
    # ---------- 预测步（闭眼，按模型猜）----------
    # 公式①: x̂⁻ = F · x̂_{k-1}    （一维，没有控制输入 u）
    x_prior_val = F * x_post_prev

    # 公式②: P⁻ = F² · P_{k-1} + Q
    P_prior_val = F * P_post_prev * F + Q

    # ---------- 更新步（睁眼，拿传感器数据修正）----------
    # 公式③: K = P⁻ / (P⁻ + R)    （一维，H=1）
    K_val = P_prior_val / (P_prior_val + R)

    # 公式④: x̂ = x̂⁻ + K(z - x̂⁻)   （后验 = 先验 + K×残差）
    x_post_val = x_prior_val + K_val * (measured[k] - x_prior_val)

    # 公式⑤: P = (1 - K) · P⁻      （不确定度缩小）
    P_post_val = (1.0 - K_val) * P_prior_val

    # ---------- 存到数组里（方便后面画图）----------
    x_prior[k]  = x_prior_val
    x_post[k]   = x_post_val
    P_prior[k]  = P_prior_val
    P_post[k]   = P_post_val
    K_values[k] = K_val

    # ---------- 打印每步的关键数据 ----------
    # x_prior = x^-, x_post = x^, P_prior = P^-, P_post = P
    print(f"k={k:2d} | "
          f"prior={x_prior_val:7.3f} | "
          f"z={measured[k]:7.3f} | "
          f"post={x_post_val:7.3f} | "
          f"P-={P_prior_val:8.3f} | "
          f"K={K_val:.4f} | "
          f"P={P_post_val:8.3f}")

    # ---------- 准备下一轮 ----------
    x_post_prev = x_post_val
    P_post_prev = P_post_val

# ============================================================
# 5. 画图 — 三张图一目了然
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# ---- 图1: 位置跟踪 ----
ax1 = axes[0]
ax1.plot(true_pos, 'k-',  linewidth=1.5, label='True position')
ax1.plot(measured,  'rx', markersize=5, alpha=0.5, label='Measurements (noisy)')
ax1.plot(x_post,   'b-', linewidth=2.0, label='KF estimate')
ax1.set_ylabel('Position (m)')
ax1.set_xlabel('Time step k')
ax1.set_title('1D Kalman Filter — Position Tracking')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ---- 图2: 卡尔曼增益 K 的变化 ----
ax2 = axes[1]
ax2.plot(K_values, 'g-', linewidth=2.0)
ax2.set_ylabel('Kalman Gain K')
ax2.set_xlabel('Time step k')
ax2.set_title('K over time: starts high (trust sensor), converges low (trust model)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# ---- 图3: 协方差 P 的变化（预测 vs 更新） ----
ax3 = axes[2]
ax3.plot(P_prior, 'r--', linewidth=1.5, label='Prior P- (after predict, rises)')
ax3.plot(P_post,  'b-',  linewidth=2.0, label='Posterior P (after update, drops)')
ax3.set_ylabel('Covariance P (uncertainty)')
ax3.set_xlabel('Time step k')
ax3.set_title('P sawtooth: predict adds Q, update pulls it back down')
#ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'kf_1d_result.png')
plt.savefig(output_path, dpi=150)
print(f"\nImage saved to: {output_path}")
plt.show()
