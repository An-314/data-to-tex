import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

# t0.95(ν)因子查询表
t_table = {
    1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
    12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04, 40: 2.02, 50: 2.01, 60: 2.00, 70: 1.99, 100: 1.98, float('Inf'): 1.96
}

# 设置图标格式
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 解析命令行参数
parser = argparse.ArgumentParser(description='进行线性最小二乘拟合并绘制图像。')
parser.add_argument('filename', type=str, help='包含数据的.txt文件名')
parser.add_argument('mode', type=int, nargs='?', default=1, choices=[1, 2, 3], help='1为y-x的线性拟合，2为lny-lnx的线性拟合，3为y-1/x的线性拟合')

args = parser.parse_args()
filename = args.filename
mode = args.mode

# 读取.txt文件
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 提取必要信息
header = lines[0].strip()
other_info = lines[1].strip()
x_line = lines[2].split(":")
x_name = x_line[0].strip()
x_data = np.array(list(map(float, x_line[1].strip().split())))
# 读取多个y轴的数据，并计算平均值
y_data_list = []
for line in lines[3:]:
    if ":" in line:
        y_data_list.append(list(map(float, line.split(":")[1].strip().split())))
y_data_array = np.array(y_data_list)
y_data = np.mean(y_data_array, axis=0)
y_name = lines[3].split(":")[0].strip()

# 错误处理
if len(x_line) < 2:
    print("[Error] 输入文件格式错误。")
    exit(1)

# 数据处理和拟合
if mode == 2:
    x_data = np.log(x_data)
    y_data = np.log(y_data)
    x_name = f'ln[{x_name}]'
    y_name = f'ln[{y_name}]'

elif mode == 3:
    x_data = 1/x_data
    x_name = f'1/{x_name}'

coefficients = np.polyfit(x_data, y_data, 1)
slope, intercept = coefficients
y_fit = slope * x_data + intercept

# 计算不确定度（如果倒数第二行有不确定度参数）
if ":" not in lines[-2]:
    delta_B = float(lines[-2].strip())
    n = len(y_data_list)  # y 的测量次数
    if n == 1:
        delta_A = 0
    else:
        v = n - 1  # 自由度
        t_value = t_table.get(v, t_table[float('inf')])  # 获取 t0.95(v)
        S_x = np.std(y_data_array, axis=0, ddof=1)  # 标准偏差
        delta_A = (t_value / np.sqrt(n)) * S_x  # 根据公式计算 delta_A
else:
    delta_A = delta_B = None
delta = np.sqrt(delta_A ** 2 + delta_B ** 2) if delta_A is not None else None
print(delta)

# 读取精确位数（从最后一行）
if " " in lines[-1]:
    precision1, precision2 = map(int, lines[-1].strip().split())
else:
    print("[Error] 输入文件格式错误。")
    exit(1)

# 使用matplotlib进行绘图
plt.figure(figsize=(10, 6))
plt.title(header)
plt.scatter(x_data, y_data, label='测量数据', marker='o')
# plt.errorbar(x_data, y_data, yerr=delta, fmt='o', label='不确定度')
plt.plot(x_data, y_fit, label=f'拟合线: y = {slope:.2f}x + {intercept:.2f}', color='red')
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.legend()
plt.figtext(0.5, 0.01, other_info, wrap=True, horizontalalignment='center', fontsize=10)

# 保存图像，文件名与输入文件相同，扩展名为.png
output_filename = os.path.splitext(filename)[0] + '.png'
plt.savefig(output_filename)
# plt.show()

# 生成Latex表格代码 #
# 生成表格
latex_table= "\\begin{table}[h]\n"
latex_table += "\\centering\n"
# 根据 n 的值决定 y 列的列数
if n > 1:
    latex_table += "\\begin{tabular}{|c|" + "c|" * n + "c|}\n"
    latex_table += "\\hline\n"
    latex_table += f"{x_name} & {' & '.join([f'{y_name}:测量{i+1}' for i in range(n)])} & {y_name}:平均值 \\\\\n"
else:
    latex_table += "\\begin{tabular}{|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += f"{x_name} & {y_name} \\\\\n"
latex_table += "\\hline\n"
# 添加数据
if n == 1:
    for x, y in zip(x_data, y_data_array[0]):
        latex_table += f"{x:.{precision1}f} & {y:.{precision2}f} \\\\\n"
else:
    for x_idx, x in enumerate(x_data):
        latex_table += f"{x:.{precision1}f} & {' & '.join([f'{y:.{precision2}f}' for y in y_data_array[:, x_idx]])}"
        # 如果有不确定度，显示平均值（±Δ）
        if delta_A is not None:
            latex_table += f" & {y_data[x_idx]:.{precision2}f} ($\\pm$ {delta[x_idx]:.{precision2}f}) \\\\\n"
        else:
            latex_table += f" & {y_data[x_idx]:.{precision2}f} \\\\\n"
latex_table += f"\\hline\n\\end{{tabular}}\n\\caption{{{header}实验的测量数据}}\\end{{table}}\n"
# 生成包含图像的Latex代码
output_filename_only = os.path.basename(os.path.splitext(filename)[0] + '.png')
latex_figure = f"\\begin{{figure}}[h]\n\\centering\n\\includegraphics[scale=0.5]{{{output_filename_only}}}\n\\caption{{{header}的拟合结果}}\n\\end{{figure}}\n"
# 将Latex代码合并，并保存到.tex文件中
latex_content = "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{grffile}\n\\usepackage{ctex}\n\\begin{document}\n"
latex_content += f"在本次实验中，我们在{other_info}条件下，探究{header}。\n"
latex_content += latex_table
if delta_A is not None:
    latex_content += f"\n由不确定度的计算公式：$\\Delta_A = \\frac{{t_p(v)}}{{\\sqrt n}} S_u$。\n"
    formatted_S_x = [f"{value:.{precision2}f}" for value in S_x]
    formatted_S_x_str = ",".join(formatted_S_x)
    latex_content += f"\n其中，$n={n}$，$v={v}$，查表得到$t_p(v)={t_value}$。通过计算，我们可以得到$S_u=\\sqrt{{\\frac{{\\sum_{{i=1}}^{{n}}(u-u_i)^2}}{{n-1}}}}={formatted_S_x_str}$。\n"
    formatted_delta_A = [f"{value:.{precision2}f}" for value in delta_A]
    formatted_delta_B = f"{delta_B:.{precision2}f}"
    formatted_delta = [f"{value:.{precision2}f}" for value in delta]
    formatted_delta_A_str = ",".join(formatted_delta_A)
    formatted_delta_B_str = formatted_delta_B
    formatted_delta_str = ",".join(formatted_delta)
    latex_content += f"\n因此，$\\Delta_A={formatted_delta_A_str}$。\n\n再根据$\\Delta_B={formatted_delta_B_str}$，我们可以得到：\n\n$\\Delta=\\sqrt{{\\Delta_A^2+\\Delta_B^2}}={formatted_delta_str}$。\n"
latex_content += "\n由这样的数据，我们可以得到拟合结果如下图所示。\n"
latex_content += latex_figure
latex_content += "\\end{document}"
with open(os.path.splitext(filename)[0] + '.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)
print("LaTeX 文件已生成。")