from mlxtend.frequent_patterns import apriori as mapriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules
import itertools
import pandas as pd


def create_one_hot_vec(df):
    items = (df['0'].unique())
    itemset = set(items)
    encoded_vals = []
    for index, row in df.iterrows():
        rowset = set(row)
        labels = {}
        uncommons = list(itemset - rowset)
        commons = list(itemset.intersection(rowset))
        for uc in uncommons:
            labels[uc] = 0
        for com in commons:
            labels[com] = 1
        encoded_vals.append(labels)
    encoded_vals[0]
    ohe_df = pd.DataFrame(encoded_vals)
    return ohe_df


def get_apriori_rules(df):
    ohe_df = create_one_hot_vec(df)
    freq_items = mapriori(ohe_df, min_support=0.1, use_colnames=True)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    return assoc_rules


def get_fpgrowth_rules(df):
    ohe_df = create_one_hot_vec(df)
    freq_items = fpgrowth(ohe_df, min_support=0.1, use_colnames=True, verbose=1)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    return assoc_rules


def get_fpmax_rules(df):
    ohe_df = create_one_hot_vec(df)
    freq_items = fpmax(ohe_df, min_support=0.1, use_colnames=True, verbose=1)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=0.1, support_only=True)
    return assoc_rules


def get_eclat_rules(df):
    parameters = {"min_support": 0.1, "max_length": 3}
    assoc_rules = df.get_eclat(params=parameters)
    return assoc_rules


def get_common_rules(left_rules, right_rules):
    common_rules = []
    for left_sub_list in left_rules:
        left_sub_list = sorted(left_sub_list)
        for rightSubList in right_rules:
            if left_sub_list == rightSubList:
                if left_sub_list not in common_rules:
                    common_rules.append(left_sub_list)

    return common_rules


def get_all_rules(left_rules, right_rules):
    new_left_rules = []
    new_right_rules = []
    for leftSubList in left_rules:
        new_left_rules.append(sorted(leftSubList))
    left_rules = list(new_left_rules for new_left_rules, _ in itertools.groupby(new_left_rules))

    whole_rules = left_rules

    for rightSubList in right_rules:
        new_right_rules.append(sorted(rightSubList))
    right_rules = list(new_right_rules for new_right_rules, _ in itertools.groupby(new_right_rules))

    for rightSubList in right_rules:
        if rightSubList not in whole_rules:
            whole_rules.append(rightSubList)

    whole_rules = sorted(whole_rules)
    return whole_rules
