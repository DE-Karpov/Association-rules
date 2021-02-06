from apyori import apriori
import pyfpgrowth
import pandas as pd
import numpy as np
import streamlit as st

class AssocRules:

    def __init__(self):
        self.dataset = pd.read_csv("../data/Basket.csv", header = None)
        self.transactions = []
        self.fill_transactions()

    def fill_transactions(self):
        for i in range(0, 7501): 
            self.transactions.append([str(self.dataset.values[i,j]) for j in range(0, 20) if not pd.isnull(self.dataset.values[i,j])])    

    class Eclat:

        def __init__(self, min_support = 0.01, max_items = 5, min_items = 2):
            self.min_support = min_support
            self.max_items = max_items
            self.min_items = min_items
            self.item_lst = list()
            self.item_len = 0
            self.item_dict = dict()
            self.final_dict = dict()
            self.data_size = 0
        
        def read_data(self, dataset):
            for index, row in dataset.iterrows():
                row_wo_na = set(row)
                for item in row_wo_na:
                    if pd.isnull(item):
                        continue
                    else:
                        item = item.strip()
                    if item in self.item_dict:
                        self.item_dict[item][0] += 1
                    else:
                        self.item_dict.setdefault(item, []).append(1)
                    self.item_dict[item].append(index)

            self.data_size = dataset.shape[0]
            self.item_lst = list(self.item_dict.keys())
            self.item_len = len(self.item_lst)
            self.min_support = self.min_support * self.data_size
            
        def recur_eclat(self, item_name, tids_array, minsupp, num_items, k_start):
            if tids_array[0] >= minsupp and num_items <= self.max_items:
                for k in range(k_start+1, self.item_len):
                    if self.item_dict[self.item_lst[k]][0] >= minsupp:
                        new_item = item_name + " | " + self.item_lst[k]
                        new_tids = np.intersect1d(tids_array[1:], self.item_dict[self.item_lst[k]][1:])
                        new_tids_size = new_tids.size
                        new_tids = np.insert(new_tids, 0, new_tids_size)
                        if new_tids_size >= minsupp:
                            if num_items >= self.min_items: self.final_dict.update({new_item: new_tids})
                            self.recur_eclat(new_item, new_tids, minsupp, num_items+1, k)
        
        def fit(self, dataset):
            i = 0
            self.read_data(dataset)
            for w in self.item_lst:
                self.recur_eclat(w, self.item_dict[w], self.min_support, 2, i)
                i+=1
            return self
            
        def transform(self):
            return {k: "{0:.4f}%".format((v[0]+0.0)/self.data_size*100) for k, v in self.final_dict.items()}


    def get_apriori(self, params):
        rules = list(apriori(self.transactions, min_support = params["min_support"][0], min_confidence = params["min_confidence"][0],
                                            min_lift = params["min_lift"][0], min_length = params["min_length"][0]))
        list_of_rules = [list(record.items) for record in rules]
        return list_of_rules

    def get_eclat(self, params):
        model = AssocRules.Eclat(min_support = params["min_support"][0], max_items = 4, min_items = params["min_length"][0])
        model.fit(self.dataset)
        return model.transform()

    def get_fpgrowth(self, params):
        patterns = pyfpgrowth.find_frequent_patterns(self.transactions, 21)
        new_patterns = {k: v for k, v in patterns.items() if not (("nan") in k)}
        return list(pyfpgrowth.generate_association_rules(new_patterns, params["min_confidence"][0]))

    def user_input_features(self):
        min_support = st.sidebar.slider("Minimal support", 0.01, 0.03, 0.001)
        min_confidence = st.sidebar.slider("Min confidence", 0.2, 0.6, 0.1)
        min_lift = st.sidebar.slider("Lift", 0.5, 6.0, 1.0)
        min_length = st.sidebar.slider("Min length", 1, 3, 1)
        data = {"min_support": min_support,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "min_length": min_length}
        features = pd.DataFrame(data, index=[0])
        return features




st.write("""
# Association rules
This app shows rules for market basket !
""")

rules = AssocRules()

df = rules.user_input_features()

st.write(df)

st.subheader("Apriori")

st.write("Rules: ", rules.get_apriori(df))

st.subheader("Eclat")

st.write("Rules: ", rules.get_eclat(df))

st.subheader("FPGrowth")

st.write("Rules: ", rules.get_fpgrowth(df))