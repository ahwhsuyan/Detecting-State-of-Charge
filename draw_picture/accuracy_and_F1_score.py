import numpy as np
import matplotlib.pyplot as plt

# 设置字体为Times New Roman，符合IEEE风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.facecolor"] = "white"

# 生成时间数据（归一化为0~1，这里代表充电过程，从0到1可认为是0到T分钟）
t = np.linspace(0, 1, 200)

# 采用logistic函数生成SOC曲线
# SOC = 100 / (1 + exp(-k*(t - 0.5)))
k = 10  # 曲线陡峭程度
SOC = 100 / (1 + np.exp(-k * (t - 0.5)))

# 定义函数，求解给定SOC对应的t（反函数公式）
def find_t_for_SOC(soc_target, k=10):
    # 对于soc_target (0<soc_target<100)，
    # 100/(1+exp(-k*(t-0.5))) = soc_target  =>  exp(-k*(t-0.5)) = (100/soc_target) - 1
    # 则 t = 0.5 - (1/k)*ln(100/soc_target - 1)
    return 0.5 - (1/k) * np.log(100/soc_target - 1)

# 定义关键SOC值及对应的t
soc_points = [0, 25, 50, 75, 100]
t_points = []
for soc in soc_points:
    if soc == 0:
        t_points.append(0)   # 对于0%，直接取t=0
    elif soc == 100:
        t_points.append(1)   # 对于100%，直接取t=1
    else:
        t_points.append(find_t_for_SOC(soc, k))

# 创建图形
fig, ax = plt.subplots(figsize=(6,4))

# 绘制SOC充电曲线
ax.plot(t, SOC, color='blue', linewidth=2, label='SOC charging curve')

# 设置坐标轴标签（IEEE风格）
ax.set_xlabel("Time (min)", fontsize=20)
ax.set_ylabel("SoC (%)", fontsize=20)

# 设置坐标轴范围和刻度
ax.set_xlim(0, 1)
ax.set_ylim(0, 110)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(['$t_0$', '$t_1$', '$t_2$', '$t_3$', '$t_4$'])  # 设置横坐标标签
ax.set_yticks([0, 25, 50, 75, 100])


ax.tick_params(axis='both', labelsize=16) 
# 添加网格，采用虚线风格
ax.grid(True, linestyle='--', linewidth=0.5)

# 隐藏上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()