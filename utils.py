import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float

    total_size = 0
    weighted_sum = 0.0

    for branch in branches:
        subtotal = 0
        for classification in branch:
            subtotal += classification
        print("total: " + str(subtotal))
        total_size += subtotal

        child_entropy = 0
        for classification in branch:
            if classification == 0:
                continue
            x = 1.0 * classification / subtotal
            child_entropy += -x * np.log2(x)
        print("child_entropy: " + str(child_entropy))

        weighted_sum += child_entropy * subtotal

    print("total size: " + str(total_size))
    return S - weighted_sum / total_size
    # return S - conditional_entropy(branches)


# def conditional_entropy(branches):
#     # branches: List[List[int]]
#     # return: float
#
#     total_size = 0
#     weighted_sum = 0.0
#
#     for branch in branches:
#         subtotal = 0
#         for classification in branch:
#             subtotal += classification
#         # print("total: " + str(subtotal))
#         total_size += subtotal
#
#         child_entropy = 0
#         for classification in branch:
#             if classification == 0:
#                 continue
#             x = 1.0 * classification / subtotal
#             child_entropy += -x * np.log2(x)
#         # print("child_entropy: " + str(child_entropy))
#
#         weighted_sum += child_entropy * subtotal
#
#     return weighted_sum / total_size


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List

    if not decisionTree.root_node.splittable:
        return

    while True:
        parent_predicted_Y = decisionTree.predict(X_test)
        parent_error = count_error(y_test, parent_predicted_Y)
        pruned_node = None
        print("total length: " + str(len(y_test)))
        print("parent error count: " + str(parent_error))

        # traverse the tree
        queue = []
        for c in decisionTree.root_node.children:
            queue.insert(0, c)

        while len(queue) > 0:
            child_node = queue.pop()
            if child_node.splittable:
                # predict
                child_node.splittable = False
                predicted_Y = decisionTree.predict(X_test)
                child_error = count_error(y_test, predicted_Y)
                if child_error < parent_error:
                    parent_error = child_error
                    pruned_node = child_node

                child_node.splittable = True

                for c in child_node.children:
                    queue.insert(0, c)

        # prune the subtree into leaf
        if pruned_node is not None:
            pruned_node.splittable = False
        else:
            break


def count_error(y_val, y_predict):
    error = 0

    for i in range(len(y_val)):
        if y_predict[i] != y_val[i]:
            error += 1

    return error


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
