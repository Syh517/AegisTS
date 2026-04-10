from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


font_path = '/home/yyy/mysoftware/fonts/times.ttf' 
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 尝试查找 Times New Roman
try:
    prop = font_manager.FontProperties(family='Times New Roman')
    file_path = font_manager.findfont(prop)
    print(f"当前指向的字体文件路径: {file_path}")
except Exception as e:
    print("没找到 Times New Roman")