import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

# from Isolation_forest_helper_funcs import train_IF, debug_tree, anomaly_detection_results_visualisation, iterate_through_tree

# TODO: add c
class SyntheticDatasetGenerator:

    def label_dataset(self, normal_distribution, samples):
        # sample 90% from normal dataset
        numbers = random.sample(list(normal_distribution), samples)

        # no for not an outlier 
        labels = ["no" for _ in numbers]

        # Create df with two columns (number, label)
        df = pd.DataFrame({'number': numbers, 'label_gt': labels})

        # Assign labels
        df["numeric_label_gt"] = np.where(df['label_gt'] == "yes", -1, 1)

        # shuffle rows
        df = df.sample(frac=1)

        return df

    def plot_histogram_of_data(self, dataset):    
        _, ax = plt.subplots(1,1,figsize=(12,8))
        sns.histplot(dataset, ax=ax, shrink=0.8, stat='count').set(title=f"Non outliers: Data distribution for mean:{dataset.mean()} and std:{dataset.std()}")
        ax.axvline(dataset.mean(), color='green', label='mean')
        ax.legend()
        plt.show()

    def custom_dataset_generation(self, mean=0.0, std=0.0, generated_samples=10000):
        normal_distribution = np.random.normal(mean, std, generated_samples)
        dst = pd.DataFrame({'number': normal_distribution})
        self.plot_histogram_of_data(normal_distribution)
        # func to label dataset
        labeled_dataset = self.label_dataset(normal_distribution, samples=generated_samples) 
        return dst