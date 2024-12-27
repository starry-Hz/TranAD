from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
        'SWaT1': [(0.993, 1), (0.9929, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
        'WADI1': [(0.92, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
        'HVAC': [(0.97, 1.02), (0.94, 1.05)],
        'els': [(0.35, 1), (0.63, 1.05)],
        'PSM': [(0.5, 1.05), (0.999991, 1.05)],
	}
# 选择合适的阈值，如果是TranAD模型，选择第二个阈值
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
        'SWaT1': 0.001, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
        'WADI1': 0.0001, 
        'PSM': 0.001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
        'HVAC': 0.0001, 
        'els': 0.0001, 
	}
# 选择合适的学习率
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
        'SWaT1': (90, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
        'WADI1': (99, 1200),
        'PSM': (99, 10),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
        'HVAC': (80, 5),
        'els': (80, 20),
	}
# percentiles 是一个字典，包含每个数据集的百分位阈值和滑动窗口大小
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9