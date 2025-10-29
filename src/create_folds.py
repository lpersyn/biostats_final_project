import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import argparse

RANDOM_SEED = 4872224

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified folds for cross-validation.")
    parser.add_argument("-X", "--features_path", type=str, help="Path to features CSV file.")
    parser.add_argument("-y", "--labels_path", type=str, help="Path to labels CSV file.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for folds.", default="./data/train/")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of folds to create.")
    args = parser.parse_args()

    X = pd.read_csv(args.features_path, index_col=0)
    X.set_index("SequencingID", inplace=True)
    X = X.drop(columns=["ModelID", "IsDefaultEntryForModel", "ModelConditionID", "IsDefaultEntryForMC"])
    if not X.index.is_unique:
        raise ValueError("Duplicate indices found in X")
    X.fillna(0, inplace=True)

    y = pd.read_csv(args.labels_path)
    y.set_index("SequencingID", inplace=True)
    y = y["Lineage"]

    combined = pd.concat([y, X], axis=1)
    combined.dropna(inplace=True)

    class_counts = combined[y.name].value_counts()
    classes_to_keep = class_counts[class_counts >= 10].index
    combined = combined[combined[y.name].isin(classes_to_keep)]

    os.makedirs(args.output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_assignments = pd.Series(index=combined.index, dtype=int)

    for fold, (_, val_index) in enumerate(skf.split(combined, combined[y.name])):
        fold_assignments.iloc[val_index] = fold

    for fold_num in range(args.num_folds):
        fold_indices = fold_assignments[fold_assignments == fold_num].index
        fold_data = combined.loc[fold_indices]
        fold_data.to_csv(os.path.join(args.output_dir, f"fold_{fold_num}.csv"))

    fold_assignments.to_csv(os.path.join(args.output_dir, "fold_assignments.csv"), header=["fold"])
