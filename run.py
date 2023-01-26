import sys
import pandas as pd
from utils import *
from mlpnas import MLPNAS
import yaml

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == "__main__":
    file_conf = "parameters.yaml"
    action = "testing" #"load", "training", "testing"
    if len(sys.argv) > 1:
        file_conf = sys.argv[1]
    if len(sys.argv) > 2:
        action = sys.argv[2]
    conf = None
    with open(file_conf, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
            conf = Struct(**conf)
        except yaml.YAMLError as exc:
            sys.exit(1)
    if action == "training":
        """
        Training the networks and dumping those in the logs folder
        """
        data = pd.read_csv(conf.CSV_TRAINING)
        x = data.drop(conf.CLASS_COLUMN, axis=1, inplace=False).values
        y = None
        if conf.IS_TARGET_CATEGORICAL:
            y = pd.get_dummies(data[conf.CLASS_COLUMN]).values
        else:
            y = data[conf.CLASS_COLUMN].values
        nas_object = MLPNAS(x, y, conf)
        data = nas_object.search()
    elif action == "plot_training":
        """
        Doing some preliminary plotting
        """
        get_top_n_architectures(conf.TOP_N, conf.TARGET_CLASSES, conf.nodes, conf.activation_functions)
        # get_accuracy_distribution()
    elif action == "load":
        nas_object = MLPNAS(None, None, conf)
        nas_object.load_from_configuration_folder(get_latest_folder())
        nas_object.load_search_result()
    elif action == "testing":
        data = pd.read_csv(conf.CSV_TESTING)
        x = data.drop(conf.CLASS_COLUMN, axis=1, inplace=False).values
        y = None
        if conf.IS_TARGET_CATEGORICAL:
            y = pd.get_dummies(data[conf.CLASS_COLUMN]).values
        else:
            y = data[conf.CLASS_COLUMN].values
        nas_object = MLPNAS(x, y, conf)
        nas_object.load_from_configuration_folder(get_latest_folder())
        print(calculate_mse_per_architecture(nas_object.make_testing_predictions(x,y)))

