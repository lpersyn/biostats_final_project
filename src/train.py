import pandas as pd
import argparse
from model import LGBM_Model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LightGBM model on the dataset.")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory containing training data folds.", default="./data/train/")
    parser.add_argument("-m", "--save_path", type=str, help="Path to save the trained model.", default="./models/")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    model = LGBM_Model(num_folds=args.num_folds, data_dir=args.data_dir, config={"n_estimators": 50, "num_leaves": 16, "verbosity": -1})
    model.fit(log_file=args.save_path + "training_log.txt")
    model.save_model(args.save_path + "lgbm_model.pkl")

