import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import IsolationForest

# from custom_dataset import generate_dataset, custom_dataset_generation, plot_histogram_of_datasets
# from isolation_forest_helper_funcs import iterate_through_tree, anomaly_detection_results_visualisation, train_IF


class PacRpad:
    def __init__(self, epsilon, delta, mu):
        self.epsilon = epsilon
        self.delta = delta
        self.mu = mu

        self.dataset = None
        self.model = None

        self.U_h = None
        self.H = None

    # def has_rare_pattern(self, x,D,H_hat,mu):
    #     '''
    #     x : datapoint
    #     D : trained set
    #     H : pattern space
    #     mu : Threshold for the estimated frequency of a pattern to be detected :  tau + epsilon/2
    #     '''
    #     U_h = calculate_U_h(x, H)

    #     rare = False

    # size_D = calculate_size_of_training_data_set(D) # |D| # (length of D)
    # TODO: revisit this part
    # for pattern_counter, h in enumerate(H):

    # 	# estimate the normalized pattern probiblites (f_hat) using the patterns h that that satistfy h(x) == 1
    # 	estimated_pattern_probability = calculate_f_hat(x, h, D, U_h)

    # 	# decision_rule: detect x as anomaleous if any estimated normalized pattern probability is smaller than mu (f_hat(h) < mu)
    # 	if estimated_pattern_probability < mu:
    # 		rare = True # (anomaly)
    # 		break

    # decision.append(estimated_pattern_probability, rare)

    # return rare

    def find_pattern_index(self, data_point, filtered_sorted_thresholds):
        pattern = -1
        for c_t, t in enumerate(filtered_sorted_thresholds):
            if data_point < t:
                print(f"dp : {data_point} satisfies pattern space: {c_t}")
                pattern_index = c_t
                break
            if data_point > t and c_t == len(filtered_sorted_thresholds) - 1:
                print(f"dp : {data_point} satisfies last pattern space")
                pattern_index = len(filtered_sorted_thresholds) + 1

        return pattern_index

    def calculate_f_hat(self, data_point, H, data_set, U_h_forest, list_of_thresholds):
        """
        x : datapoint
        H : set of pattern spaces.
        |D| : size of training set.
        U_h : The area of the axis aligned hyper rectangle (pattern) in the pattern space divided by the area of the biggest one.
        list_of_thresholds : list_of_thresholds: contains Thresholds for split nodes in ExtraRegressorTree for each tree.
        """
        f_hat_list = []
        indeces = []
        print(H)
        print(U_h_forest)
        for tree, _ in enumerate(H):
            pattern_index = find_pattern_index(data_point, list_of_thresholds[tree])
            f_hat_tree = H[tree][pattern_index] / (
                len(data_set) * U_h_forest[tree][pattern_index]
            )
            f_hat_list.append(f_hat_tree)

        f_hat = np.min(f_hat_list)
        index_min_f = np.argmin(f_hat_list)

        return f_hat

    def calculate_U_h_tree(self, volumes_pattern_space_tree):
        total_volume = np.sum(volumes_pattern_space_tree)
        U_h = volumes_pattern_space_tree / total_volume
        return U_h

    def calculate_U_h_forest(self, volumes_pattern_space_forest):
        """
        x : datapoint
        h : trained set
        |D| : size of training set?
        U_h : The area of the rectangle (pattern) in the pattern space
        """
        U_h_forest = []
        for pattern_space in np.arange(0, len(volumes_pattern_space_forest), 1):
            U_h_forest.append(
                calculate_U_h_tree(volumes_pattern_space_forest[pattern_space])
            )
        return U_h_forest


def main():
    sim = PacRpad(epsilon=0.1, delta=0.1, mu=0.1)

    # dataset =  custom_dataset_generation(mean=15,std=1,generated_samples=10000)
    # model, dataset_post_detection = train_IF(dataset, n_estimators=2)
    # anomaly_detection_results_visualisation(dataset_post_detection)


if __name__ == "__main__":
    main()
