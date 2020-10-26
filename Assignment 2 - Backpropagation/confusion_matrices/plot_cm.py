import numpy as np
# For plotting confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# Evaluate performance on test dataset, plot confusion matrix
cm = np.genfromtxt('1_test.csv', dtype='d', delimiter=',').astype(int).transpose()

df_cm = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
print(df_cm)
plt.figure(figsize=(10,7))
hmap = sn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True, fmt='d')
plt.xlabel("Targets")
plt.ylabel("Predictions")
plt.axis('tight')

plt.show()