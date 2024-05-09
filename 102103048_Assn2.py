import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def topsis(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    nm = a / np.sqrt(np.sum(a**2, axis=0))
    wnm = nm * b
    isolution = np.max(wnm, axis=0) if c else np.min(wnm, axis=0)
    aisolution = np.min(wnm, axis=0) if c else np.max(wnm, axis=0)
    separationfromi = np.sqrt(np.sum((wnm - isolution)**2, axis=1))
    separationfromai = np.sqrt(np.sum((wnm - aisolution)**2, axis=1))
    closeness = separationfromai / (separationfromi + separationfromai)
    rankings = np.argsort(closeness)[::-1] + 1
    return closeness, rankings

# Read the new data from the updated CSV file
data = pd.read_csv(r'C:\Users\91896\Desktop\Topsis-Chandan-102103047\Topsis on Text Summarization\assign_2.csv')

# Apply TOPSIS using the 'Accuracy', 'AP', and 'F1' columns
closeness, rankings = topsis(data[['ACCURACY', 'AP', 'F1']], [1, 1, 1], True)

# Add 'Closeness' and 'Rankings' columns to the DataFrame
data['Closeness'] = closeness
data['Rankings'] = rankings

# Print the updated DataFrame
print(data)

# Plotting the bar graph
plt.figure(figsize=(10, 5))
plt.bar(data.index, data['Closeness'], color='blue', label='Closeness')
plt.bar(data.index, data['Rankings'], color='orange', label='Rankings')
plt.xlabel('Data Point')
plt.ylabel('Values')
plt.title('Closeness and Rankings for TOPSIS Evaluation')
plt.legend()
plt.show()
