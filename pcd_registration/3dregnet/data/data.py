
from registration.global_registration import prepare_dataset

#######################################################
def load_data():
    print('Loading test data...')
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(0.05)
    data = {}
    data["x1"] = source_fpfh
    data["x2"] = target_fpfh
    data["sd"] = source_down
    data["td"] = target_down
    return data