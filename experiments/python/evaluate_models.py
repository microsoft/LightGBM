import os
import re
import matplotlib.pyplot as plt

def GetAccuracyFromLightGBM(filename, key):
	input = open(filename, "r")
	ret = 0.0
	for line in input.readlines():
		if key + ' :' in line:
			ret = float(line.split(key+' :')[-1])
	return ret

def GetValueFromTXT(filename, key):
	input = open(filename, "r")
	ret = 0.0
	for line in input.readlines():
		if key + '=' in line:
			if key == 'num_leaves':
				ret += float(line.split(key+'=')[-1])
			else:
				ret = float(line.split(key+'=')[-1])
		elif key + ': ' in line and key != 'num_leaves':
			ret = float(line.split(key+': ')[-1][:-2])
	return ret

def plotMetrics(keyword):
	sorted_dir = sorted(os.listdir("../results"), key=lambda x: int(x.split(".")[1]))
	setting_value = []
	accuracies = []
	no_features = []
	no_thresholds = []
	no_leaves = []
	no_trees = []
	for fn in sorted_dir:
		if fn.endswith(keyword+".out"):
			accuracies.append(GetAccuracyFromLightGBM('../results/'+fn, 'auc'))
		if fn.endswith(keyword+".txt"):
			no_trees.append(GetValueFromTXT('../results/'+fn, 'Tree'))
			setting_value.append(GetValueFromTXT('../results/'+fn, keyword))
			no_features.append(GetValueFromTXT('../results/'+fn, 'tt_feature_count'))
			no_thresholds.append(GetValueFromTXT('../results/'+fn, 'tt_threshold_count'))
			no_leaves.append(GetValueFromTXT('../results/'+fn, 'num_leaves'))
		else:
			continue
	fig1, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel(keyword)
	ax1.set_ylabel('AUC')
	ax1.plot(setting_value, accuracies, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis


	ax2.set_ylabel('count')  # we already handled the x-label with ax1
	ax2.plot(setting_value, no_thresholds, label="no. thresholds")
	ax2.plot(setting_value, no_leaves, label="no. leaves")
	plt.legend(loc='lower right')

	color = 'tab:green'
	ax3 = ax1.twinx()  # instantiate a third Axes that shares the same x-axis
	ax3.tick_params(axis='y', labelcolor=color)
	ax3.plot(setting_value, no_features, label="no. features", color=color)

	fig1.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.legend(loc='upper left')
	plt.show()

plotMetrics('num_iterations')
plotMetrics('max_depth')
plotMetrics('tinygbdt_forestsize')

