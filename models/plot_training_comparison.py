#!/usr/bin/env python3
"""
Plot training performance comparison from two CSV files with Gaussian smoothing.
Usage: python plot_training_comparison.py <filename1> <label1> <filename2> <label2> <title>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def gaussian_smooth(data, sigma=30):
    """Apply Gaussian smoothing to data."""
    return ndimage.gaussian_filter1d(data, sigma=sigma)

def plot_training_comparison(filename1, label1, filename2, label2, title):
    """Plot training performance comparison with raw data and Gaussian smoothed trendlines."""
    
    try:
        df1 = pd.read_csv(filename1)
        df2 = pd.read_csv(filename2)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    batches1 = df1['batch'].values
    rewards1 = df1['average_reward'].values
    
    batches2 = df2['batch'].values
    rewards2 = df2['average_reward'].values
    
    smoothed_rewards1 = gaussian_smooth(rewards1, sigma=30)
    smoothed_rewards2 = gaussian_smooth(rewards2, sigma=30)
    
    last_10_percent1 = int(len(rewards1) * 0.1)
    if last_10_percent1 > 0:
        avg_trained_reward1 = np.mean(rewards1[-last_10_percent1:])
    else:
        avg_trained_reward1 = rewards1[-1]
    
    last_10_percent2 = int(len(rewards2) * 0.1)
    if last_10_percent2 > 0:
        avg_trained_reward2 = np.mean(rewards2[-last_10_percent2:])
    else:
        avg_trained_reward2 = rewards2[-1]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(batches1, rewards1, color='orange', alpha=0.4, linewidth=1, label=f'{label1} - Raw Data')
    plt.plot(batches1, smoothed_rewards1, color='orange', linewidth=2, label=f'{label1} - Gaussian Trend (σ=30)')
    
    plt.plot(batches2, rewards2, color='green', alpha=0.4, linewidth=1, label=f'{label2} - Raw Data')
    plt.plot(batches2, smoothed_rewards2, color='green', linewidth=2, label=f'{label2} - Gaussian Trend (σ=30)')
    

    plt.title(title, fontsize=16)
    plt.xlabel('Training Batch', fontsize=12)
    plt.ylabel('Average Reward Per Episode', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    legend = plt.legend(loc='lower right', fancybox=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    plt.show()

def main():
    if len(sys.argv) != 6:
        print("Usage: python plot_training_comparison.py <filename1> <label1> <filename2> <label2> <title>")
        print("Example: python plot_training_comparison.py file1.csv 'Method A' file2.csv 'Method B' 'Training Comparison'")
        sys.exit(1)
    
    filename1 = sys.argv[1]
    label1 = sys.argv[2]
    filename2 = sys.argv[3]
    label2 = sys.argv[4]
    title = sys.argv[5]
    
    plot_training_comparison(filename1, label1, filename2, label2, title)

if __name__ == "__main__":
    main()