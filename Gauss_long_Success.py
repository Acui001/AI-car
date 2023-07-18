import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# 设置均值和标准差
mu = 0.5
sigma = 0.18
area_number=7
# 定义积分函数
def integral_function(x):
    return gaussian(x, mu, sigma)


# 进行积分计算
result = integrate.quad(integral_function, 0, 1)
area_total = result[0]  # 获取积分结果，面积总和
area_per_interval = area_total / area_number  # 计算每个区间应有的面积

# 寻找使得累计面积达到目标的区间划分点
x = np.linspace(0, 1, 10000)  # 用更多的点进行积分计算以提高精度
cumulative_area = np.zeros_like(x)
current_area = 0
interval_points = []

for i in range(len(x)):
    result = integrate.quad(integral_function, 0, x[i])
    cumulative_area[i] = result[0]  # 获取积分结果，累计面积
    if cumulative_area[i] >= current_area:
        interval_points.append(x[i])
        current_area += area_per_interval


# 绘制高斯分布函数和区间划分点
y = gaussian(x, mu, sigma)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gaussian Distribution')
plt.grid(True)

for point in interval_points:
    plt.axvline(point, color='r', linestyle='--')

plt.show()

# 打印区间列表
print(interval_points)
weights=[]
total_weight = 0

# 计算总面积来为下面计算以保证归一化
for i in range(area_number):
    weight = round(interval_points[i + 1] - interval_points[i], 4)
    total_weight += weight

for i in range(area_number):
    weight = round((interval_points[i + 1] - interval_points[i]) / total_weight, 4)
    normalized_weight = round(weight , 4)
    print(interval_points[i + 1], '-', interval_points[i], '=', normalized_weight)
    weights.append(normalized_weight)

print(weights)  # [0.3094, 0.0894, 0.069, 0.0644, 0.069, 0.0894, 0.3093]
