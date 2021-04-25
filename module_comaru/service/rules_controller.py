from service.get_rules import get_common_rules, get_all_rules, get_apriori_rules, get_fpgrowth_rules, \
    get_fpmax_rules, get_eclat_rules


def get_all_common_rules(data):
    list_of_prepared_rules = prepare_rules(data)
    common_rules_app_growth = get_common_rules(list_of_prepared_rules[0], list_of_prepared_rules[1])
    common_rules_max_eclat = get_common_rules(list_of_prepared_rules[2], list_of_prepared_rules[3])
    common_rules = get_common_rules(common_rules_app_growth, common_rules_max_eclat)
    return common_rules


def get_all_assoc_rules(data):
    list_of_prepared_rules = prepare_rules(data)
    all_rules_app_growth = get_all_rules(list_of_prepared_rules[0], list_of_prepared_rules[1])
    all_rules_max_eclat = get_all_rules(list_of_prepared_rules[2], list_of_prepared_rules[3])
    all_rules = get_all_rules(all_rules_app_growth, all_rules_max_eclat)
    return all_rules


def transform(df):
    list_of_rules = []
    for item in df:
        intermediate_rules = [next(iter(item[0])), next(iter(item[1]))]
        list_of_rules.append(intermediate_rules)
    return list_of_rules


def prepare_rules(data):
    prepared_apriori_data = transform(
        get_apriori_rules(data).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_fpgrowth_data = transform(
        get_fpgrowth_rules(data).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_fpmax_data = transform(get_fpmax_rules(data).reset_index()[['antecedents', 'consequents']].values.tolist())
    prepared_eclat_data = get_eclat_rules(data)
    return prepared_apriori_data, prepared_fpgrowth_data, prepared_fpmax_data, prepared_eclat_data
