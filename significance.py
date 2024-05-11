import scipy.stats as stats

# Define the datasets
first_data = [0.887109, 0.652168, 0.615365, 0.643812, 0.703023, 0.692474, 0.664794, 0.649053, 0.629006, 0.614247, 0.605598, 0.608312, 0.602658, 0.587594, 0.589158, 0.586408, 0.576668]
second_data = [0.887109, 0.651911, 0.621538, 0.649343, 0.697056, 0.675252, 0.656409, 0.642474, 0.61827, 0.609561, 0.601547, 0.596397, 0.583517, 0.585795, 0.582329, 0.572212, 0.562908]
third_data = [0.886749, 0.649287, 0.578476, 0.530747, 0.622002, 0.593804, 0.583826, 0.528462, 0.522178, 0.524233, 0.481976, 0.526019, 0.496095, 0.542027, 0.484296, 0.512035, 0.484955]

# Perform Kruskal-Wallis test
H, p = stats.kruskal(first_data, second_data, third_data)
print("Kruskal-Wallis Test:")
print("H-statistic:", H)
print("p-value:", p)

# Perform pairwise comparisons using Dunn's test with Bonferroni correction
# You may need to install the scikit-posthocs library for the following code
from scikit_posthocs import posthoc_dunn

data = [first_data, second_data, third_data]
posthoc_results = posthoc_dunn(data, p_adjust='bonferroni')
print("\nPairwise Post-hoc Tests:")
print(posthoc_results)

import scipy.stats as stats

# Define the datasets
first_data = [0.887109, 0.652168, 0.615365, 0.643812, 0.703023, 0.692474, 0.664794, 0.649053, 0.629006, 0.614247, 0.605598, 0.608312, 0.602658, 0.587594, 0.589158, 0.586408, 0.576668]
second_data = [0.887109, 0.651911, 0.621538, 0.649343, 0.697056, 0.675252, 0.656409, 0.642474, 0.61827, 0.609561, 0.601547, 0.596397, 0.583517, 0.585795, 0.582329, 0.572212, 0.562908]
third_data = [0.886749, 0.649287, 0.578476, 0.530747, 0.622002, 0.593804, 0.583826, 0.528462, 0.522178, 0.524233, 0.481976, 0.526019, 0.496095, 0.542027, 0.484296, 0.512035, 0.484955]

# Perform the Wilcoxon rank-sum test
statistic, p_value = stats.ranksums(first_data, second_data)
print("Wilcoxon Rank-Sum Test (First vs. Second):")
print("Test Statistic:", statistic)
print("p-value:", p_value)

statistic, p_value = stats.ranksums(first_data, third_data)
print("\nWilcoxon Rank-Sum Test (First vs. Third):")
print("Test Statistic:", statistic)
print("p-value:", p_value)

statistic, p_value = stats.ranksums(second_data, third_data)
print("\nWilcoxon Rank-Sum Test (Second vs. Third):")
print("Test Statistic:", statistic)
print("p-value:", p_value)

import scipy.stats as stats

# Define the datasets
first_data = [0.887109, 0.652168, 0.615365, 0.643812, 0.703023, 0.692474, 0.664794, 0.649053, 0.629006, 0.614247, 0.605598, 0.608312, 0.602658, 0.587594, 0.589158, 0.586408, 0.576668]
second_data = [0.887109, 0.651911, 0.621538, 0.649343, 0.697056, 0.675252, 0.656409, 0.642474, 0.61827, 0.609561, 0.601547, 0.596397, 0.583517, 0.585795, 0.582329, 0.572212, 0.562908]
third_data = [0.886749, 0.649287, 0.578476, 0.530747, 0.622002, 0.593804, 0.583826, 0.528462, 0.522178, 0.524233, 0.481976, 0.526019, 0.496095, 0.542027, 0.484296, 0.512035, 0.484955]

# Perform the independent t-test
t_statistic, p_value = stats.ttest_ind(first_data, second_data)
print("Independent t-test (First vs. Second):")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

t_statistic, p_value = stats.ttest_ind(first_data, third_data)
print("\nIndependent t-test (First vs. Third):")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

t_statistic, p_value = stats.ttest_ind(second_data, third_data)
print("\nIndependent t-test (Second vs. Third):")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
