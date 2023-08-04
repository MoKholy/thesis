import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# define function to split our data into train, labels and string labels
def split_data(data):
    x = data.iloc[:, :-2]
    y_string = data.iloc[:, -2]
    y = data.iloc[:, -1]
    return x, y, y_string


# plot 2d data
def plot_2d_data(x, class_labels, title):

    # colour map
    class_colors = {
    'SSH-Bruteforce': 'red',
    'Benign': 'blue',
    'DDoS attacks-LOIC-HTTP': 'green',
    'DDOS attack-HOIC': 'orange',
    'DoS attacks-Slowloris': 'purple',
    'DoS attacks-Hulk': 'brown',
    'FTP-BruteForce': 'pink',
    'Infilteration': 'gray',
    'Bot': 'cyan',
    'DoS attacks-GoldenEye': 'magenta',
    'Brute Force -Web': 'yellow',
    'DoS attacks-SlowHTTPTest': 'teal',
    'SQL Injection': 'lime',
    'DDOS attack-LOIC-UDP': 'salmon',
    'Brute Force -XSS': 'olive'
    }

    # colours
    colors = [class_colors[label] for label in class_labels]

    # plot
    plt.figure(figsize=(8, 6))

    # plot data
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.title(title)
    plt.show()