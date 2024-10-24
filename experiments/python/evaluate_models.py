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
			ret = float(line.split(key+'=')[-1])
	return ret

accuracies = []
trees = []
no_features = []
no_thresholds = []

sorted_ids = []

print(os.listdir("../results"))
sorted_dir = sorted(os.listdir("../results"), key=lambda x: int(x.split(".")[1]))
print(sorted_dir)

for fn in sorted_dir:
    if fn.endswith(".out"): 
        accuracies.append(GetAccuracyFromLightGBM('../results/'+fn, 'auc'))
    if fn.endswith(".txt"):
        trees.append(GetValueFromTXT('../results/'+fn, 'Tree'))
        no_features.append(GetValueFromTXT('../results/'+fn, 'tt_feature_count'))
        no_thresholds.append(GetValueFromTXT('../results/'+fn, 'tt_threshold_count'))
    else:
        continue

print(accuracies)
print(trees)

# plt.plot(trees, accuracies, label = 'AUC')
# plt.plot(trees, no_features, label = 'no. features')
# plt.plot(trees, no_thresholds, label = 'no. thresholds')
# plt.xlabel("No. of trees")
# plt.ylabel("AUC")
# plt.legend()
# plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('No of trees')
ax1.set_ylabel('AUC')
ax1.plot(trees, accuracies, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('count')  # we already handled the x-label with ax1
ax2.plot(trees, no_features, label="no. features" )
ax2.plot(trees, no_thresholds, label="no. thresholds")
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()