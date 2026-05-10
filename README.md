# AegisTS

本仓库实现论文 **AegisTS: A Hierarchical Agent System with Reinforcement Learning for Multivariate Time Series Data Cleaning**。

- 论文链接: https://arxiv.org/html/2605.04902v2

## 项目结构与论文模块对应关系

- Error_Injection/injector.py
	- 实现数据质量问题(data quality issues)的注入逻辑, 用于生成带噪/异常的多变量时间序列数据。
	- 主要功能包括: 在样本上按比例注入单点异常、漂移、噪声、波动、渐变、突变、缺失和重复等错误类型。

- Error_Detection/
	- 实现论文中的 **Data Quality Detector** 模块。
	- 包含检测器配置、模型包装、评估与运行脚本等, 用于发现时间序列中的数据质量问题。

- Error_Cleaner/
	- 实现论文中的 **Cleaning Pipeline Generator** 模块。
	- 包含清洗策略搜索、强化学习清洗器、评估与运行脚本等, 用于生成并执行清洗流程。

