

def invert_stratification_mapping(label_to_ids):
    assert sum(map(len, label_to_ids.values())) == len(set().union(*map(set, label_to_ids.values()))), \
        'Some ids have more than one stratification label.'

    id_to_label = dict()
    for label, ids in label_to_ids.items():
        id_to_label.update({id_: label for id_ in ids})

    return id_to_label
