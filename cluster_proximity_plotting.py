import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

infile = pd.read_csv(r'C:\Users\mirio\research\cluster_proximity\test_ckdtree_10K_parallel_analysis_analysis.csv')
# print(f'file length: {len(infile)}')
# print(infile.head)
sns.set(color_codes=True)
# plot the histogram of "how many subjects"
sns.distplot(infile.how_many_subjects,kde=False, rug=True).set_title('Number of subjects distribution (out of 81)', fontsize = 14)

oneh_K = r'C:\Users\mirio\research\filtered_data_sets\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_FILTERED_DATA_10K_per_subject.csv'
infile_10K_data = pd.read_csv(oneh_K)
HC_precentage = infile_10K_data[infile_10K_data['labels']=='HC'].count()/len(infile_10K_data)

g = sns.jointplot(x='how_many_subjects', y="HC", data=infile, kind="kde", color="m")
g.plot_joint(plt.scatter, c="b", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_title('HC distribution among clusters')
g.set_axis_labels("Number of subjects in cluster", "HC");
                 
# ======= Classification results plotting
class_scores = pd.read_csv(r'C:\Users\mirio\research\logs\scores_celiac_10K_method2_290420_expanded_scores.txt')
sns.catplot(x = 'classification_method', y = 'Accuracy_score', data = class_scores, kind = 'bar')
sns.catplot(x = 'classification_method', y = 'f1_score', data = class_scores, kind = 'bar')
#p = class_scores.pivot('classification_method')
data_for_heatmap = class_scores.drop(['Inputfile', 'model_name', 'dimension_reduction_method',
       'number_of_dimensions'] , axis = 1)
data_for_heatmap = data_for_heatmap.set_index('classification_method')
fig, ax = plt.subplots(figsize=(9,9))
ax = sns.heatmap(data_for_heatmap, annot = True, linewidths = 0.25, vmin = 0, vmax = 1, robust = True, cmap="YlGnBu", ax=ax)
ax.set_title('Classification scores for different methods', fontsize = 14)
