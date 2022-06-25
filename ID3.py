import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from DecisonTree import unique_vals
from utils import *
import numpy as np
"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        # create a np array of the number of each type of label
        # for each label calculate the entropy using the equaiton provided in lecture
        values = np.array(list(counts.values()))
        sum = np.sum(values)
        values = values / sum
        values = -1 * values * np.log2(values)
        impurity = np.sum(values)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        total = len(left) + len(right)
        weighted_left = (len(left)/total) * self.entropy(left, left_labels)
        weighted_right = (len(right)/total) * self.entropy(right, right_labels)
        info_gain_value = current_uncertainty - weighted_right - weighted_left
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        # for row, label in zip(rows, labels):
        #     if question.match(row):
        #         row = np.reshape(np.array(row), (1,len(row)))
        #         if type(true_rows) is np.ndarray:
        #             true_rows = np.append(true_rows, row, axis=0)
        #             true_labels = np.append(true_labels, label)
        #         else:
        #             true_rows = row
        #             true_labels = np.append(true_labels, label)
        #     else:
        #         row = np.reshape(np.array(row), (1,len(row)))
        #         if type(false_rows) is np.ndarray:
        #             false_rows = np.append(false_rows, row, axis=0)
        #             false_labels = np.append(false_labels, label)
        #         else:
        #             false_rows = row
        #             false_labels = np.append(false_labels, label)
        # gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)
        # ========================
        for row, label in zip(rows, labels):
            if question.match(row): 
                true_rows.append(row)
                true_labels.append(label)
            else:
                false_rows.append(row)
                false_labels.append(label)
        true_rows = np.reshape(true_rows,(len(true_labels),30))
        true_labels = np.reshape(true_labels,(len(true_labels),))
        false_rows = np.reshape(false_rows,(len(false_labels),30))
        false_labels = np.reshape(false_labels,(len(false_labels),))
        gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        for colidx in range(rows.shape[1]):
            unique_values = unique_vals(rows, colidx)
            for val in unique_values:
                question = Question(None, colidx, val)
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, question, current_uncertainty)
                if gain >= best_gain:
                    best_gain = gain
                    best_question = question
                    best_false_rows, best_false_labels = false_rows, false_labels
                    best_true_rows, best_true_labels = true_rows, true_labels
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        counts = class_counts(rows, labels)
        # check if we have only one type of classification in our labels, if so return a leaf
        if len(list(counts.keys())) == 1: #if all the keys are the same, they all have the same label, return leaf
            return Leaf(rows, labels)
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)
        if len(best_false_rows) == 0: #if there are no false examples, return leaf for false branch not sure its needed
            false_branch = Leaf(best_false_rows, best_false_labels)
        else: #if we are here, we need to continure splitting the examples, in order to get better results.
            false_branch = self.build_tree(best_false_rows, best_false_labels)
        
        if len(best_true_rows) == 0: #if there are no true examples, return leaf for true branch
            true_branch = Leaf(best_true_rows, best_true_labels)
        else: #if we are here, we need to continure splitting the examples, in order to get better results.
            true_branch = self.build_tree(best_true_rows, best_true_labels)

        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def build_pruned_tree(self, rows, labels, m):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        #not sure about the split
        counts = class_counts(rows, labels)
        if len(list(counts.keys())) <= 1: #if all the keys are the same, they all have the same label, return leaf
            return Leaf(rows, labels)
        if rows.shape[0] <= m:
            return Leaf(rows, labels)
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)
        if len(best_false_rows) == 0: #if there are no false examples, return leaf for false branch not sure its needed
            false_branch = Leaf(best_false_rows, best_false_labels)
        else: 
            false_branch = self.build_pruned_tree(best_false_rows, best_false_labels,m)
        
        if len(best_true_rows) == 0: #if there are no true examples, return leaf for true branch
            true_branch = Leaf(best_true_rows, best_true_labels)
        else: 
            true_branch = self.build_pruned_tree(best_true_rows, best_true_labels,m)

        # ========================

        return DecisionNode(best_question, true_branch, false_branch)



    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def pruned_fit(self, x_train, y_train,M):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_pruned_tree(x_train, y_train,M)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        while type(node) != Leaf:
            if node.question.match(row):
                node = node.true_branch
            else:
                node = node.false_branch
        counts_dict = node.predictions
        sum = np.sum(list(counts_dict.values()))
        for  label ,counts in counts_dict.items():
            if counts >= sum/2:
                prediction = label    
        # ========================

        return prediction

    def predict(self, rows):
        """ 
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.apply_along_axis(self.predict_sample, 1, rows)
        # ========================

        return y_pred
