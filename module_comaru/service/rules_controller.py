from service.get_rules import get_common_rules, get_all_rules, get_apriori_rules, get_fpgrowth_rules, \
    get_fpmax_rules, get_eclat_rules
import itertools


def get_all_common_rules(data, params):
    support = float(params.support)
    confidence = float(params.confidence)
    list_of_prepared_rules = prepare_rules(data, support, confidence)
    rules_count = get_rules_count_number(list_of_prepared_rules)
    common_rules_apr_growth = get_common_rules(list_of_prepared_rules[0], list_of_prepared_rules[1])
    common_rules_max_eclat = get_common_rules(list_of_prepared_rules[2], list_of_prepared_rules[3])
    common_rules = get_common_rules(common_rules_apr_growth, common_rules_max_eclat)
    print_recall(rules_count, common_rules, 'intersection')
    return common_rules


def get_all_assoc_rules(data, params):
    support = float(params.support)
    confidence = float(params.confidence)
    list_of_prepared_rules = prepare_rules(data, support, confidence)
    rules_count = get_rules_count_number(list_of_prepared_rules)
    all_rules_app_growth = get_all_rules(list_of_prepared_rules[0], list_of_prepared_rules[1])
    all_rules_max_eclat = get_all_rules(list_of_prepared_rules[2], list_of_prepared_rules[3])
    all_rules = get_all_rules(all_rules_app_growth, all_rules_max_eclat)
    print_recall(rules_count, all_rules, 'union')
    return all_rules


def get_rules_count_number(rules):
    apriori_rules_count = len(get_rules_count(rules[0]))
    fpgrowth_rules_count = len(get_rules_count(rules[1]))
    fpmax_rules_count = len(get_rules_count(rules[2]))
    eclat_rules_count = len(get_rules_count(rules[3]))
    return [apriori_rules_count, fpgrowth_rules_count, fpmax_rules_count, eclat_rules_count]


def get_rules_count(rules):
    sorted_rules = []
    prepared_rules = []
    for subList in rules:
        sorted_rules.append(sorted(subList))
    for elem in sorted_rules:
        if elem not in prepared_rules:
            prepared_rules.append(elem)
    return prepared_rules


def transform(df):
    list_of_rules = []
    for item in df:
        intermediate_rules = [next(iter(item[0])), next(iter(item[1]))]
        list_of_rules.append(intermediate_rules)
    sorted_rules = []
    for subList in list_of_rules:
        sorted_rules.append(sorted(subList))
    prepared_rules = list(sorted_rules for sorted_rules, _ in itertools.groupby(sorted_rules))
    return prepared_rules


def prepare_rules(data, support, confidence):
    prepared_apriori_data = transform(
        get_apriori_rules(data, support, confidence).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_fpgrowth_data = transform(
        get_fpgrowth_rules(data, support, confidence).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_fpmax_data = transform(
        get_fpmax_rules(data, support, confidence).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_eclat_data = get_eclat_rules(data, support)
    return prepared_apriori_data, prepared_fpgrowth_data, prepared_fpmax_data, prepared_eclat_data


def print_recall(list_of_rules, rules, zone):
    rules_count = len(rules)
    apriori_rules = list_of_rules[0]
    fpgrowth_rules = list_of_rules[1]
    fpmax_rules = list_of_rules[2]
    eclat_rules = list_of_rules[3]
    if zone == 'intersection':
        print("Common rules count: ", rules_count)
        print("Apriori rules count: {}. Recall = {}".format(apriori_rules,
                                                            rules_count / apriori_rules))
        print("FP-Growth rules count: {}. Recall = {}".format(fpgrowth_rules,
                                                              rules_count / fpgrowth_rules))
        print(
            "FP-Max rules count: {}. Recall = {}".format(fpmax_rules, rules_count / fpmax_rules))
        print(
            "ECLAT rules count: {}. Recall = {}".format(eclat_rules, rules_count / eclat_rules))
    else:
        print("All rules count: ", rules_count)
        print("Apriori rules count: {}. Recall = {}".format(apriori_rules,
                                                            1 - apriori_rules / rules_count))
        print("FP-Growth rules count: {}. Recall = {}".format(fpgrowth_rules,
                                                              1 - fpgrowth_rules / rules_count))
        print(
            "FP-Max rules count: {}. Recall = {}".format(fpmax_rules, 1 - fpmax_rules / rules_count))
        print(
            "ECLAT rules count: {}. Recall = {}".format(eclat_rules, 1 - eclat_rules / rules_count))
