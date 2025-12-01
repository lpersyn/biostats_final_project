import pandas as pd

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./reports/performance/accuracy_by_lineage.csv")

    support_acc_corr = df["support"].corr(df["mean_fold_accuracy"])
    print(f"Correlation between support and accuracy: {support_acc_corr}")

    acc_std_corr = df["std_fold_accuracy"].corr(df["mean_fold_accuracy"])
    print(f"Correlation between std and accuracy: {acc_std_corr}")