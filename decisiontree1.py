from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련세트
print(dt.score(test_scaled, test_target)) # 테스트세트

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1. filled=True, feature_names=['alohol','sugar','pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_acaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input,test_target))

plt.figure(figsize=(20,15))
plot_tree(dt,filled=True, feature_names['alcohol','sugar','pH'])
plt.show()

print(dt.feature_importances_)
