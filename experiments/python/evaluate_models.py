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

accuracies_t = []
accuracies_d = []
trees = []
max_trees = []
max_depth = []
no_features_t = []
no_thresholds_t = []
no_features_d = []
no_thresholds_d = []
no_leaves_t = []
no_leaves_d = []

sorted_ids = []

print(os.listdir("../results"))
sorted_dir = sorted(os.listdir("../results"), key=lambda x: int(x.split(".")[1]))
print(sorted_dir)

for fn in sorted_dir:
	if fn.endswith("trees.out"):
		accuracies_t.append(GetAccuracyFromLightGBM('../results/'+fn, 'auc'))
	if fn.endswith("depth.out"):
		accuracies_d.append(GetAccuracyFromLightGBM('../results/'+fn, 'auc'))
	if fn.endswith("trees.txt"):
		trees.append(GetValueFromTXT('../results/'+fn, 'Tree'))
		max_trees.append(GetValueFromTXT('../results/'+fn, 'num_iterations'))
		no_features_t.append(GetValueFromTXT('../results/'+fn, 'tt_feature_count'))
		no_thresholds_t.append(GetValueFromTXT('../results/'+fn, 'tt_threshold_count'))
		no_leaves_t.append(GetValueFromTXT('../results/'+fn, 'num_leaves'))
	if fn.endswith("depth.txt"):
		max_depth.append(GetValueFromTXT('../results/'+fn, 'max_depth'))
		no_features_d.append(GetValueFromTXT('../results/'+fn, 'tt_feature_count'))
		no_thresholds_d.append(GetValueFromTXT('../results/'+fn, 'tt_threshold_count'))
		no_leaves_d.append(GetValueFromTXT('../results/'+fn, 'num_leaves'))
	else:
		continue

print(trees)

# plt.plot(trees, accuracies, label = 'AUC')
# plt.plot(trees, no_features, label = 'no. features')
# plt.plot(trees, no_thresholds, label = 'no. thresholds')
# plt.xlabel("No. of trees")
# plt.ylabel("AUC")
# plt.legend()
# plt.show()


fig1, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Max trees')
# TODO: actual no. of trees
ax1.set_ylabel('AUC')
ax1.plot(max_trees, accuracies_t, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('count')  # we already handled the x-label with ax1
ax2.plot(max_trees, no_features_t, label="no. features" )
ax2.plot(max_trees, no_thresholds_t, label="no. thresholds")
ax2.plot(max_trees, no_leaves_t, label="no. leaves")
# ax2.tick_params(axis='y', labelcolor=color)

fig1.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()


fig2, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Max depth')
ax1.set_ylabel('AUC')
ax1.plot(max_depth, accuracies_d, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('count')  # we already handled the x-label with ax1
ax2.plot(max_depth, no_features_d, label="no. features" )
ax2.plot(max_depth, no_thresholds_d, label="no. thresholds")
ax2.plot(max_depth, no_leaves_d, label="no. leaves")
# ax2.tick_params(axis='y', labelcolor=color)

fig2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()