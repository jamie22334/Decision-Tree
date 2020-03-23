import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        min_entropy = np.inf
        max_feature = -1
        dimension = len(self.features[0])
        # print("dimension: " + str(dimension) + " features[0]: " + str(self.features[0]))
        features_array = np.array(self.features)
        # print("features_array: " + str(features_array))

        # find dimension of feature that leads to the smallest conditional entropy
        for dim in range(dimension):
            feature_map = dict()
            label_map = dict()
            unique_features = np.unique(features_array[:, dim])
            unique_labels = np.unique(self.labels)

            F = len(unique_features)
            L = len(unique_labels)
            # print("unique_features: " + str(unique_features))
            # print("F: " + str(F))
            # print("L: " + str(L))

            # map[feature] = index
            for i in range(F):
                feature_map[unique_features[i]] = i

            for i in range(L):
                label_map[unique_labels[i]] = i
            # print(feature_map)
            # print(label_map)

            branches = np.zeros(F * L).reshape(F, L)
            for i, feature in enumerate(self.features):
                branches[feature_map[feature[dim]]][label_map[self.labels[i]]] += 1
            # print(branches)

            cur_entropy = self.conditional_entropy(branches)
            # print("cur_entropy: " + str(cur_entropy) + " min_entropy: " + str(min_entropy))
            if cur_entropy < min_entropy:
                min_entropy = cur_entropy
                self.dim_split = dim
                self.feature_uniq_split = sorted(unique_features)
                max_feature = F
                # print("split at dim: " + str(dim))
            # handle tie!!
            elif cur_entropy == min_entropy:
                if F > max_feature:
                    self.dim_split = dim
                    self.feature_uniq_split = sorted(unique_features)
                    max_feature = F
                    # print("split at dim: " + str(dim))

        if min_entropy == np.inf:
            self.splittable = False
            return

        # split at dim_split and add children
        # child_node_list = list()
        for value in self.feature_uniq_split:
            child_features = list(list())
            child_labels = list()

            for i, feature in enumerate(self.features):
                if feature[self.dim_split] == value:
                    reduced_feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
                    child_features.append(reduced_feature)
                    child_labels.append(self.labels[i])
            child_num_cls = len(np.unique(child_labels))
            # handle children order
            self.children.append(TreeNode(child_features, child_labels, child_num_cls))

        # split child nodes recursively
        for c in self.children:
            if c.splittable:
                c.split()

    def conditional_entropy(self, branches):
        # branches: List[List[int]]
        # return: float

        total_size = 0
        weighted_sum = 0.0

        for branch in branches:
            subtotal = 0
            for classification in branch:
                subtotal += classification
            # print("total: " + str(subtotal))
            total_size += subtotal

            child_entropy = 0
            for classification in branch:
                if classification == 0:
                    continue
                x = 1.0 * classification / subtotal
                child_entropy += -x * np.log2(x)
            # print("child_entropy: " + str(child_entropy))

            weighted_sum += child_entropy * subtotal

        return weighted_sum / total_size

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int

        if self.splittable:
            # print("feature: " + str(feature))
            child_index = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            # print("after remove: " + str(feature))
            return self.children[child_index].predict(feature)
        else:
            return self.cls_max
