# AegisTS

This repository implements the paper **AegisTS: A Hierarchical Agent System with Reinforcement Learning for Multivariate Time Series Data Cleaning**.

- Paper link: https://arxiv.org/html/2605.04902v2

## Project Structure and Paper Module Mapping

- Error_Injection/injector.py
	- Implements data quality issue injection logic for generating noisy/anomalous multivariate time series.

- Error_Detection/
	- Implements the **Data Quality Detector** module from the paper.

- Error_Cleaner/
	- Implements the **Cleaning Pipeline Generator** module from the paper.

