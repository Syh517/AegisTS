# AegisTS

This repository implements the paper **AegisTS: A Hierarchical Agent System with Reinforcement Learning for Multivariate Time Series Data Cleaning**.

- Paper link: https://arxiv.org/html/2605.04902v2

## Project Structure and Paper Module Mapping

- Error_Injection/injector.py
	- Implements data quality issue injection logic for generating noisy/anomalous multivariate time series.
	- Supports injecting single-point anomalies, drift, Gaussian noise, volatility, gradual change, sudden change, missing values, and duplicates.

- Error_Detection/
	- Implements the **Data Quality Detector** module from the paper.
	- Contains detector configuration, model wrappers, evaluation, and run scripts to identify data quality issues.

- Error_Cleaner/
	- Implements the **Cleaning Pipeline Generator** module from the paper.
	- Contains cleaning strategy search, RL-based cleaners, evaluation, and run scripts to generate and execute cleaning pipelines.

