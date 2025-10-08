import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datasets.tools import set_times_new_roman_font, save_figure

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

set_times_new_roman_font()


def get_data(file_path):
    df = pd.read_csv(file_path)
    values = df['Value'].values
    return values.tolist()


total_loss_data = get_data(r"total_loss.csv")
val_loss_data = get_data(r"val_loss.csv")
learning_rate_data = get_data(r"learning_rate.csv")

fig, ax = plt.subplots(figsize=(12.8, 7.0), constrained_layout=True)

font_size = 18
x_label = 'Iterations'
y_label = 'Loss of training data'

ax.plot(range(len(total_loss_data)), total_loss_data, color='blue', linewidth=2)

ax.set_xlabel(x_label, fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.legend(fontsize=font_size)
ax.tick_params(labelsize=font_size)

# Keep existing grid style; no change to tick spacing
ax.grid(True, linestyle='--', alpha=0.5)

save_figure('', "total_loss")


fig, ax = plt.subplots(figsize=(12.8, 7.0), constrained_layout=True)

font_size = 18
x_label = 'Iterations'
y_label = 'Loss of validation data'

ax.plot([5 * i for i in range(len(val_loss_data))], val_loss_data, color='blue', linewidth=2)

ax.set_xlabel(x_label, fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.legend(fontsize=font_size)
ax.tick_params(labelsize=font_size)

# Keep existing grid style; no change to tick spacing
ax.grid(True, linestyle='--', alpha=0.5)

save_figure('', "val_loss")

print(len(learning_rate_data))  # 745
print(len(total_loss_data))  # 1730


def interpolate_list(data, target_len):
    if not data:
        return []
    if len(data) == target_len:
        return data
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_len)
    y_new = np.interp(x_new, x_old, data)
    return y_new.tolist()


interpolated_learning_rate_data = interpolate_list(learning_rate_data, len(total_loss_data))
print(len(interpolated_learning_rate_data))  # 1730

fig, ax = plt.subplots(figsize=(12.8, 7.0), constrained_layout=True)

font_size = 18
x_label = 'Iterations'
y_label = 'Learning rate'

ax.plot(range(len(learning_rate_data)), learning_rate_data, color='blue', linewidth=2)

ax.set_xlabel(x_label, fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.legend(fontsize=font_size)
ax.tick_params(labelsize=font_size)

# Keep existing grid style; no change to tick spacing
ax.grid(True, linestyle='--', alpha=0.5)

save_figure('', "learning_rate")
