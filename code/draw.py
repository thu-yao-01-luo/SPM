import numpy as np
import matplotlib.pyplot as plt
import os

# Dictionary to store accuracy results
accuracy_dict = {}

# Load numpy matrices from .npy files
file_list = [file for file in os.listdir('.') if file.endswith('.npy')]
for file_name in file_list:
    matrix = np.load(file_name)

    # Compute accuracy
    trace = np.trace(matrix)
    matrix_sum = np.sum(matrix)
    accuracy = trace / matrix_sum

    # Store accuracy in dictionary
    accuracy_dict[file_name] = accuracy

# Draw histogram
fig, ax = plt.subplots()
ax.bar(range(len(accuracy_dict)), list(accuracy_dict.values()), align='center')
ax.set_xticks(range(len(accuracy_dict)))
ax.set_xticklabels(list(accuracy_dict.keys()), rotation=45)
ax.set_xlabel('File Name')
ax.set_ylabel('Accuracy')
print(accuracy_dict)
plt.savefig('accuracy.png')
# plt.show()
