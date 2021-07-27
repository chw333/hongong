import numpy as np
sc = SGDClassfier(loss='log', random_state=42)
train_score = []
test_score=[]
classes = np.unique(train_traget)

for _ in range(0,300):
  sc.partial_fit(train_scaled, train_target, classes_classes)
  train_score.append(sc.score(rain_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_taget))
  
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

sc = SGDClassifer(loss='log', max_iter=100, tol=N?one, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.SGDClassfier(loss='hinge', max_iter=100, tol=None, randome_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
