#!/usr/bin/env python3
"""
Plot training performance from CSV files with Gaussian smoothing.
Usage: python plot_training.py <filename> <title>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def gaussian_smooth(data, sigma=30):
    """Apply Gaussian smoothing to data."""
    return ndimage.gaussian_filter1d(data, sigma=sigma)

def plot_training_performance(filename, title):
    """Plot training performance with raw data and Gaussian smoothed trendline."""
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    batches = df['batch'].values
    rewards = df['average_reward'].values
    
    smoothed_rewards = gaussian_smooth(rewards, sigma=30)
    
    last_10_percent = int(len(rewards) * 0.1)
    if last_10_percent > 0:
        avg_trained_reward = np.mean(rewards[-last_10_percent:])
    else:
        avg_trained_reward = rewards[-1]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(batches, rewards, color='orange', alpha=0.4, linewidth=1, label='Raw Data')
    plt.plot(batches, smoothed_rewards, color='orange', linewidth=2, label='Gaussian Trend (σ=30)')
    
    plt.axhline(y=avg_trained_reward, color='gray', linestyle='--', linewidth=2, label=f'Average Trained Reward: {avg_trained_reward:.1f}')
    
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
    if len(sys.argv) != 3:
        print("Usage: python plot_training.py <filename> <title>")
        print("Example: python plot_training.py actor_critic/actor_critic_training_reward_per_batch.csv 'Actor-Critic Training Performance'")
        sys.exit(1)
    
    filename = sys.argv[1]
    title = sys.argv[2]
    
    plot_training_performance(filename, title)

if __name__ == "__main__":
    main()