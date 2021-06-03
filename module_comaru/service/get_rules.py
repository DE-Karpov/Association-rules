from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules
import pandas as pd

from algos import ECLAT


def create_one_hot_vec(df):
    items = (df['0'].unique())
    itemset = set(items)
    encoded_vals = []
    for index, row in df.iterrows():
        rowset = set(row)
        rowset = {element for element in rowset if pd.notna(element)}
        labels = {}
        uncommons = list(itemset - rowset)
        commons = list(itemset.intersection(rowset))
        for uc in uncommons:
            labels[uc] = 0
        for com in commons:
            labels[com] = 1
        encoded_vals.append(labels)
    ohe_df = pd.DataFrame(encoded_vals)
    return ohe_df


def get_apriori_rules(df, support, confidence):
    ohe_df = create_one_hot_vec(df)
    freq_items = apriori(ohe_df, min_support=support, use_colnames=True)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
    return assoc_rules


def get_fpgrowth_rules(df, support, confidence):
    ohe_df = create_one_hot_vec(df)
    freq_items = fpgrowth(ohe_df, min_support=support, use_colnames=True, verbose=1)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
    return assoc_rules


def get_fpmax_rules(df, support, confidence):
    ohe_df = create_one_hot_vec(df)
    freq_items = fpmax(ohe_df, min_support=support, use_colnames=True, max_len=5, verbose=1)
    assoc_rules = association_rules(freq_items, metric="confidence", min_threshold=confidence, support_only=True)
    return assoc_rules


def get_eclat_rules(df, support):
    params = {"min_support": support, "max_length": 5}
    model = ECLAT.Eclat(min_support=params["min_support"], max_items=params["max_length"], min_items=1)
    model.fit(df)
    return model.transform()


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
    raw_rules = [left_rules, right_rules]
    return sort_and_distinct(raw_rules)


def sort_and_distinct(rules):
    sorted_rules = []
    prepared_rules = []
    for subList in rules:
        sorted_rules.append(sorted(subList))
    for subList in sorted_rules:
        rules_set = set(tuple(x) for x in subList)
        prepared_rules = [list(x) for x in rules_set]

    return prepared_rules
