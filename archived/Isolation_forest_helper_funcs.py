from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score, make_scorer

# TODO: Add class and inheret from ExtraTreeRegressor sklearn
def train_IF(dataset, n_estimators=50, contamination=0.1, max_features=1.0):

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
    )
    model.fit(dataset[["number"]])

    dataset["scores"] = model.decision_function(dataset[["number"]])
    dataset["anomaly"] = model.predict(dataset[["number"]])
    dataset["anomaly"] = np.where(dataset["anomaly"] == -1, "yes", "no")

    outliers = dataset.loc[dataset["anomaly"] == "yes"]
    outlier_index = list(outliers.index)

    return model, dataset


class PatternSpaceIsolationForest:
    def generate_axis_aligned_hyper_rectangles_forest(self, model, dataset):
        volumes_pattern_space_forest = []
        samples_forest = []
        thresholds_forest = []

        cmap = plt.get_cmap("gnuplot")
        min_value = np.min(dataset["number"].values)
        max_value = np.max(dataset["number"].values)
        OFFSET = max_value - min_value

        for p in range(len(model.estimators_)):

            indices_samples_tree = model.estimators_samples_[p]
            samples_tree = dataset["number"].values[indices_samples_tree]

            (
                is_leaves,
                node_depth,
                children_left,
                children_right,
                _,
                thresholds,
            ) = self.iterate_through_tree(model, tree_number=p)

            print("This is the thresholds list:", thresholds)

            colors = [cmap(i) for i in np.linspace(0, 1, len((thresholds)))]
            plt.figure(figsize=(20, 10))
            plt.hlines(
                1,
                min_value - (max_value - min_value) - 0.5,
                max_value + (max_value - min_value) + 0.5,
            )

            sorted_thresholds = np.sort(np.unique(thresholds))
            filtered_sorted_thresholds = np.delete(
                sorted_thresholds, np.where(sorted_thresholds == -2)
            )
            print("sorted_thresholds:", filtered_sorted_thresholds)

            plt.eventplot(
                [filtered_sorted_thresholds[0] - OFFSET],
                orientation="horizontal",
                colors=colors[-1],
                label=f"fake_threshold first = {filtered_sorted_thresholds[0]-OFFSET}",
            )
            plt.eventplot(
                [filtered_sorted_thresholds[-1] + OFFSET],
                orientation="horizontal",
                colors=colors[-2],
                label=f"fake_threshold last = {filtered_sorted_thresholds[-1]+OFFSET}",
            )

            for i in range(0, len(thresholds), 1):
                if is_leaves[i] == False:
                    print("split node with threshold: ", thresholds[i])
                    plt.eventplot(
                        [thresholds[i]],
                        orientation="horizontal",
                        colors=colors[i],
                        label=f"threshold {i} = {thresholds[i]} in IF depth: {node_depth[i]}",
                    )

            volume_pattern_space_tree = []
            # use conv hull to set fake thresholds for the half spaces on the deges
            for c in range(len(filtered_sorted_thresholds) + 1):
                if c == 0 or c == len(filtered_sorted_thresholds):
                    v = max_value - min_value
                else:
                    v = (
                        filtered_sorted_thresholds[c]
                        - filtered_sorted_thresholds[c - 1]
                    )
                print("Pattern Space volume:", v)
                volume_pattern_space_tree.append(v)

            samples_tree_y = [1 for s in samples_tree]
            plt.plot(
                samples_tree,
                samples_tree_y,
                "x",
                color="g",
                label="value used for split",
            )
            plt.xlabel("sampled points", labelpad=7)
            plt.title(f"IF Tree {p}")
            plt.legend(loc="best")
            plt.show()

            print("volume_pattern_space_tree: ", volume_pattern_space_tree)
            volumes_pattern_space_forest.append(volume_pattern_space_tree)
            samples_forest.append(samples_tree)
            thresholds_forest.append(filtered_sorted_thresholds)

        return samples_forest, volumes_pattern_space_forest, thresholds_forest

    # TODO: change this function to take only the pattern space of the tree
    def populate_pattern_space_tree(
        self, ds1, model, tree_number, volumes_pattern_space_forest
    ):

        print(
            f"tree: {tree_number} -> Number of axis aligned hyper rectangles {len(volumes_pattern_space_forest[tree_number])}"
        )

        number_of_patterns_in_single_space = len(
            volumes_pattern_space_forest[tree_number]
        )
        patterns_occurences = np.zeros(number_of_patterns_in_single_space)

        # print("data points: ",datapoints)

        thresholds = model.estimators_[tree_number].tree_.threshold
        sorted_thresholds = np.sort(np.unique(thresholds))

        # -2 is the value assigned to undefined fields in the ExtraRegressorTree (for thresholds and features)
        filtered_sorted_thresholds = np.delete(
            sorted_thresholds, np.where(sorted_thresholds == -2)
        )
        print("filtered_sorted_thresholds: ", filtered_sorted_thresholds)
        datapoints = ds1["number"].values
        for c_dp, dp in enumerate(datapoints):
            for c_t, t in enumerate(filtered_sorted_thresholds):
                if dp < t:
                    patterns_occurences[c_t] += 1
                    # print(f"dp : {dp} in space {c_t}")
                    break
                if dp > t and c_t == len(filtered_sorted_thresholds) - 1:
                    patterns_occurences[-1] += 1
                    # print(f"dp : {dp} in space {c_t+1}")
                    break

        if np.sum(patterns_occurences) != len(datapoints):
            print("problem: not all points assigned to pattern spaces")

        print("h_list one tree:", patterns_occurences)

        return patterns_occurences, filtered_sorted_thresholds

    def populate_pattern_space_forest(self, ds1, model, volumes_pattern_space_forest):

        print("number of trees (pattern spaces): ", len(volumes_pattern_space_forest))

        number_of_estimators = len(model.estimators_)

        tree_patterns_occurences_list = []
        list_of_thresholds = []

        for tree, _ in enumerate(np.arange(0, number_of_estimators, 1)):
            (
                tree_patterns_occurences,
                filtered_sorted_thresholds,
            ) = self.populate_pattern_space_tree(
                ds1, model, tree, volumes_pattern_space_forest
            )
            tree_patterns_occurences_list.append(tree_patterns_occurences)
            list_of_thresholds.append(filtered_sorted_thresholds)

        return tree_patterns_occurences_list, list_of_thresholds

    def debug_tree(self, model, dataset, tree_number=0):
        first_tree = model.estimators_[tree_number]
        plot_tree(first_tree, impurity=False)
        plt.show()

        n_nodes = first_tree.tree_.node_count
        children_left = first_tree.tree_.children_left
        children_right = first_tree.tree_.children_right
        feature = first_tree.tree_.feature
        thresholds = first_tree.tree_.threshold
        indices_samples_tree = model.estimators_samples_[tree_number]
        samples_tree = dataset["number"].values[indices_samples_tree]

        # print("feature_names_in_:",first_tree.tree_.feature_names_in_)
        # print("dataset['number']:",dataset['number'])
        # print("parameters:",first_tree.get_params())
        # print("number of nodes in Tree: ",n_nodes)
        # print("id of the left child of node i or -1 if leaf node: left children: ",children_left)
        # print("id of the right child of node i or -1 if leaf node: right children: ",children_right)
        # print("feature used for split: ",feature)
        # print("thresholds",thresholds)

        is_leaves, node_depth, _, _, _, _ = self.iterate_through_tree(
            model, tree_number
        )

        print(
            f"The binary tree structure has {n_nodes} nodes and has the following tree structure:"
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node ".format(
                        space=node_depth[i] * "\t", node=i
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=thresholds[i],
                        right=children_right[i],
                    )
                )

        return

    def anomaly_detection_results_visualisation(self, df):
        sns.set(rc={"figure.figsize": (11.7, 8.27)})

        plt.plot(df["number"], df["scores"], ".")
        plt.title("datapoints and predicted anomaly score")
        plt.xlabel("datapoint (number)")
        plt.ylabel("anomaly score")
        outliers = df[df["anomaly"] == "yes"]
        non_outliers = df[df["anomaly"] == "no"]

        plt.figure()
        plt.title("outliers and non outliers")
        outliers_data = pd.DataFrame(
            {"number": outliers["number"], "scores": outliers["scores"]}
        )
        sns.scatterplot(
            x="number", y="scores", data=outliers_data, palette="red", label="outliers"
        )

        normal_data = pd.DataFrame(
            {"number": non_outliers["number"], "scores": non_outliers["scores"]}
        )
        sns.scatterplot(
            x="number",
            y="scores",
            data=normal_data,
            palette="green",
            label="non outliers",
        )
        plt.show()

    def iterate_through_tree(self, model, tree_number):

        n_nodes = model.estimators_[tree_number].tree_.node_count
        children_left = model.estimators_[tree_number].tree_.children_left
        children_right = model.estimators_[tree_number].tree_.children_right
        features = model.estimators_[tree_number].tree_.feature
        thresholds = model.estimators_[tree_number].tree_.threshold

        # For example, the arrays feature and threshold only apply to split nodes
        # print("features:", feature)

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)

        # start with the root node id (0) and its depth (0)
        stack = [(0, 0)]
        rectangle_edges = []

        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]

            # If a split node, append left and right children and depth to `stack` so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        return (
            is_leaves,
            node_depth,
            children_left,
            children_right,
            features,
            thresholds,
        )
