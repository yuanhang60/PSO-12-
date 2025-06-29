# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:04:26 2025

@author: wyh03
"""

# 整合后的完整代码：基于PSO的增透膜设计（12层膜，TiO2/SiO2交替）

import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------
# 材料折射率模型函数定义
# -------------------------
def n_SiO2(lambda_nm):
    λ_um = lambda_nm * 1e-3
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
    n_sq = 1 + B1*λ_um**2/(λ_um**2 - C1) \
            + B2*λ_um**2/(λ_um**2 - C2) \
            + B3*λ_um**2/(λ_um**2 - C3)
    return np.sqrt(n_sq)

A_tio2, B_tio2 = 2.15454545, 0.0712727273
def n_TiO2(lambda_nm):
    λ_um = lambda_nm * 1e-3
    return A_tio2 + B_tio2/(λ_um**2)

def n_glass(lambda_nm):
    return 1.52

# -------------------------
# 传输矩阵法计算透过率
# -------------------------
def calc_transmittance(thicknesses, wavelengths):
    n0 = 1.0
    ns = 1.52
    T_vals = np.zeros_like(wavelengths, dtype=float)
    for idx, λ in enumerate(wavelengths):
        M11 = 1+0j; M12 = 0+0j
        M21 = 0+0j; M22 = 1+0j
        for j, d in enumerate(thicknesses):
            if j % 2 == 0:
                n_layer = n_SiO2(λ)
            else:
                n_layer = n_TiO2(λ)
            delta = 2 * np.pi * n_layer * d / λ
            cosδ = np.cos(delta)
            sinδ = np.sin(delta)
            η = n_layer
            M11_layer = cosδ
            M12_layer = 1j * sinδ / η
            M21_layer = 1j * η * sinδ
            M22_layer = cosδ
            M11_new = M11 * M11_layer + M12 * M21_layer
            M12_new = M11 * M12_layer + M12 * M22_layer
            M21_new = M21 * M11_layer + M22 * M21_layer
            M22_new = M21 * M12_layer + M22 * M22_layer
            M11, M12, M21, M22 = M11_new, M12_new, M21_new, M22_new
        t = 2 * n0 / (n0*M11 + n0*ns*M12 + M21 + ns*M22)
        T_vals[idx] = abs(t)**2 * (ns / n0).real
    avg_T = T_vals.mean()
    return T_vals, avg_T

# -------------------------
# PSO优化参数与初始化
# -------------------------
num_particles = 30
num_iterations = 1000
w = 0.7
c1 = 1.5
c2 = 1.5
thickness_min = 20.0
thickness_max = 150.0
wavelengths = np.linspace(400, 700, 61)

particles = []
velocities = []
pbest_positions = []
pbest_fitness = []
gbest_position = None
gbest_fitness = -1.0

for _ in range(num_particles):
    position = np.array([random.uniform(thickness_min, thickness_max) for _ in range(12)])
    velocity = np.array([random.uniform(-(thickness_max-thickness_min)*0.1, 
                                       (thickness_max-thickness_min)*0.1) for _ in range(12)])
    _, fitness = calc_transmittance(position, wavelengths)
    pbest_positions.append(position.copy())
    pbest_fitness.append(fitness)
    if fitness > gbest_fitness:
        gbest_fitness = fitness
        gbest_position = position.copy()
    particles.append(position)
    velocities.append(velocity)

# -------------------------
# PSO 主循环
# -------------------------
for t in range(num_iterations):
    for i in range(num_particles):
        _, fitness = calc_transmittance(particles[i], wavelengths)
        if fitness > pbest_fitness[i]:
            pbest_fitness[i] = fitness
            pbest_positions[i] = particles[i].copy()
        if fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest_position = particles[i].copy()
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()
        velocities[i] = w * velocities[i] \
                        + c1 * r1 * (pbest_positions[i] - particles[i]) \
                        + c2 * r2 * (gbest_position - particles[i])
        particles[i] = particles[i] + velocities[i]
        for j in range(12):
            if particles[i][j] < thickness_min:
                particles[i][j] = thickness_min
                velocities[i][j] *= -1
            elif particles[i][j] > thickness_max:
                particles[i][j] = thickness_max
                velocities[i][j] *= -1

# -------------------------
# 输出最优结果并绘图
# -------------------------
best_thicknesses = gbest_position
T_spectrum, avg_T = calc_transmittance(best_thicknesses, np.linspace(400, 700, 301))

print("最优膜层厚度结构 (nm):", np.round(best_thicknesses, 2))
print("400-700nm 平均透过率: {:.2f}%".format(avg_T * 100))

plt.figure(figsize=(8, 5))
plt.plot(np.linspace(400, 700, 301), T_spectrum, label='Transmittance')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmittance')
plt.title('Optimized AR Coating Transmittance (12 layers)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 输出为 TFCalc 膜系文件格式
# -------------------------
def export_to_tfcalc(filename, thicknesses):
    materials = []
    for i in range(len(thicknesses)):
        mat = 'SiO2' if i % 2 == 0 else 'TiO2'
        materials.append(mat)

    with open(filename, 'w') as f:
        f.write("; Optimized multilayer AR coating\n")
        f.write("; Format: [Material] [Thickness in nm]\n")
        f.write("Glass\n")  # Substrate
        for mat, thick in zip(materials, thicknesses):
            f.write(f"{mat:<8s} {thick:.2f}\n")
        f.write("Air\n")  # Incident medium

    print(f"\nTFCalc 文件已生成: {filename}")

# 调用导出函数
export_to_tfcalc("optimized_structure.TFD", best_thicknesses)

