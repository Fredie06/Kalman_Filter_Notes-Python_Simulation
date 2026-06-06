"""
从0开始python的学习，因为没有时间去完整的学习语法，所以这里就从比较常用的numpy库和matplotlib库开始学习
仿照AI给的一维卡尔曼滤波我们来写一个二维的，也就是位置和速度

这里的观测器我们只观测位置
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#在程序开始的时候需要定义的重要参数
DT = 0.1 
MAX_STEP = 100 
Init_Velocity = 2.0 #物体始终保持匀速直线运动
Init_Position = 0
x_init_array = np.array([[Init_Position,Init_Velocity]]).T

#我们需要调节的超参数,这里的Q的计算用到了概率论的东西，我看不懂就让AI给我写了
Q = np.array([[DT**4/4, DT**3/2],
              [DT**3/2, DT**2]]) * 1
R = 100

#用于打印图表的超长向量

true_pos = np.zeros(MAX_STEP) #我们永远无法得知的物体的真实位置
measured = np.zeros(MAX_STEP) #传感器回传的有噪声的数据，方差是R
x_prior  = np.zeros(MAX_STEP) #先验估计值的物体位置
x_post   = np.zeros(MAX_STEP) #后验估计值也就是最优秀估计值的物体位置
K_pos_hist = np.zeros(MAX_STEP)   # 记录每步位置增益
K_vel_hist = np.zeros(MAX_STEP)   # 记录每步速度增益

np.random.seed(42)  # 固定随机种子，每次运行结果一样（调试用）

for k in range(MAX_STEP):        # for (k = 0; k < TOTAL_STEPS; k++)
    # 真实位置 = 初始位置 + 速度 × 时间
    true_pos[k] = Init_Position+Init_Velocity * (k * DT)

    # 传感器读数 = 真实位置 + 噪声（模拟传感器不准）
    # np.random.randn() 返回一个服从 N(0,1) 的随机数
    # sqrt(R) × N(0,1) = N(0, R)
    noise = np.sqrt(R) * np.random.randn()
    measured[k] = true_pos[k] + noise

#一些重要的矩阵
H = np.array([[1,0]])
F = np.array([[1,DT],
              [0,1]])
P_Init_matrix = np.array([[1000,0],
                          [0,1000]])
x_best_prev_array = x_init_array
P_best_prev_matrix = P_Init_matrix

print("-"*45)
print("KF Main Loop")
print("=" * 70)

for i in range(MAX_STEP):
    #以下主要就是根据五个核心公式进行的预测和更新的全部步骤，没有什么神秘的
    x_prior_array = F@x_best_prev_array

    p_prior_matrix = F@P_best_prev_matrix@(F.T)+Q

    Kalman = (p_prior_matrix@(H.T))/((H@p_prior_matrix@(H.T))+R)

    x_post_array = x_prior_array + Kalman*(measured[i]-H@x_prior_array)

    p_post_matrix = (np.eye(2)- Kalman@H)@(p_prior_matrix)
   #更新数据下一次循环用就好了
    x_best_prev_array = x_post_array
    P_best_prev_matrix = p_post_matrix

    x_prior[i] = x_prior_array[0,0]     # 存起来画图用
    x_post[i]  = x_post_array[0,0]

    K_pos_hist[i]=Kalman[0,0]
    K_vel_hist[i]=Kalman[1,0]
    print(f"i={i:2d} | "
          f"prior={x_prior[i]:7.3f} | "
          f"z={measured[i]:7.3f} | "
          f"post={x_post[i]:7.3f} | "
          f"K_pos={Kalman[0,0]:.4f} | "
          f"K_vel={Kalman[1,0]:.4f}")


fig, axes = plt.subplots(3, 1, figsize=(12, 8))

  # -- 图1: 位置跟踪 --
ax1 = axes[0]
ax1.plot(true_pos, 'k-',  linewidth=1.5, label='True position')
ax1.plot(measured,  'rx', markersize=5, alpha=1.0, label='Measurements (noisy)')
ax1.plot(x_post,    'b-', linewidth=2.0, label='KF estimate (posterior)')
ax1.plot(x_prior,   'g--', linewidth=1.0, label='KF prior (model only)')
ax1.set_ylabel('Position (m)')  
ax1.set_title('2D Kalman Filter — Position Tracking')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

  # -- 图2: 速度估计（用后验位置差分近似）--
vel_est = np.diff(x_post) / DT      # 速度 ≈ 位置变化 / 时间步长
ax2 = axes[1]
ax2.axhline(y=Init_Velocity, color='k', linestyle='-', linewidth=1.5, label='True velocity')
ax2.plot(range(1, MAX_STEP), vel_est, 'b-', linewidth=2.0, label='KF velocity estimate')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_xlabel('Time step k')
ax2.set_title('Velocity Estimate (from position difference)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(K_pos_hist,'b-',linewidth=1.5,label='Kalman_Pos')
ax3.plot(K_vel_hist,'r-',linewidth=1.5,label='Kalman_Vel')
ax3.axhline(y=1,color='g',linestyle='-',linewidth=3.0,label='1')
ax3.set_xlabel('Time step k')
ax3.set_title('kalmans')
ax3.grid(True,alpha=0.3)
# 保存图片到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'kf_2d_result.png')
plt.savefig(output_path, dpi=150)
print(f"\nImage saved to: {output_path}")
plt.tight_layout()
plt.show()
