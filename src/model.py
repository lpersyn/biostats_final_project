import lightgbm as LGBM
import pandas as pd
import pickle
import numpy as np
import tqdm
import time
import os

TEST_FOLD = 4

class LGBM_Model:
    def __init__(self, num_folds: int, data_dir:str="./data/train/", config:dict={}):
        self.num_folds = num_folds
        self.test_fold = TEST_FOLD
        self.train_val_folds = set(range(num_folds)) - {self.test_fold}
        self.data_dir = data_dir
        self.config = config
        self.models = {fold: LGBM.LGBMClassifier(**config) for fold in self.train_val_folds}
        self.genes_used_by_model = {}
        self.classes, self.num_classes = self._get_class_info()
    
    def _get_class_info(self):
        data_frames = []
        for fold in range(self.num_folds):
            df = pd.read_csv(f"{self.data_dir}/fold_{fold}.csv", index_col=0)
            data_frames.append(df)
        combined_data = pd.concat(data_frames)
        y = pd.Categorical(combined_data.pop("Lineage"))
        classes = y.categories
        num_classes = len(classes)
        return classes, num_classes

    @staticmethod
    def load_model(load_path: str):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            init_params = data.get("init_params", data.get("config", {}))
            model = LGBM_Model(**init_params)
            model.models = data["models"]
            model.genes_used_by_model = data.get("genes_used_by_model", {})
            if not model.genes_used_by_model:
                model._recompute_gene_sets()
            classes = data.get("classes")
            if classes is not None:
                model.classes = classes
                model.num_classes = len(classes)
            return model
        
    def save_model(self, save_path: str):
        # save all hyperparameter configurations and models
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "init_params": {
                    "num_folds": self.num_folds,
                    "data_dir": self.data_dir, 
                    "config": self.config
                },
                "models": self.models,
                "genes_used_by_model": self.genes_used_by_model,
                "classes": self.classes
                }, f)

    def get_train_data(self, folds):
        data_frames = []
        for fold in folds:
            df = pd.read_csv(f"{self.data_dir}/fold_{fold}.csv", index_col=0)
            data_frames.append(df)
        combined_data = pd.concat(data_frames)
        y = pd.Categorical(combined_data.pop("Lineage"))
        X = combined_data
        return X, y
    
    def reduce_gene_set(self, fold, train_X):
        gene_variances = train_X.var().sort_values(ascending=False)
        top_genes = gene_variances.head(int(0.1 * train_X.shape[1])).index
        self.genes_used_by_model[fold] = top_genes
        return train_X[top_genes]

    def _recompute_gene_sets(self):
        for fold in self.models.keys():
            train_folds = self.train_val_folds - {fold}
            train_X, _ = self.get_train_data(train_folds)
            self.reduce_gene_set(fold, train_X)

    def get_fold_data(self, fold: int, subset_to_model: bool = True, include_labels: bool = True):
        if fold not in set(range(self.num_folds)):
            raise ValueError(f"Fold {fold} is out of range for {self.num_folds} total folds.")
        df = pd.read_csv(f"{self.data_dir}/fold_{fold}.csv", index_col=0)
        y = None
        if "Lineage" in df.columns:
            y = pd.Categorical(df.pop("Lineage"), categories=self.classes)
        X = df
        if subset_to_model:
            genes = self.genes_used_by_model.get(fold)
            if genes is None:
                raise ValueError(f"Gene set for fold {fold} is not available. Train or load a model with stored gene subsets.")
            X = X.loc[:, genes]
        if include_labels and y is not None:
            return X, y
        if include_labels:
            raise ValueError("Labels were requested but not found in the fold data.")
        return X
    
    def fit(self, log_file = None):
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            log = open(log_file, "w")
        for fold, model in self.models.items():
            start_time = time.time()
            print(f"Fold {fold}:", end=" ")
            if log_file:
                log.write(f"Fold {fold}: ")
            val_data = pd.read_csv(f"{self.data_dir}/fold_{fold}.csv", index_col=0)
            val_y = pd.Categorical(val_data.pop("Lineage"), categories=self.classes)
            train_folds = self.train_val_folds - {fold}
            train_X, train_y = self.get_train_data(train_folds)
            train_X = self.reduce_gene_set(fold, train_X)
            val_data = val_data[self.genes_used_by_model[fold]]
            model.fit(train_X, train_y.codes, eval_set=[(val_data, val_y.codes)])
            acc = model.score(train_X, train_y.codes)  # type: ignore[attr-defined]
            print(f"Train acc: {acc:.4f},", end=" ")
            if log_file:
                log.write(f"Train acc: {acc:.4f}, ")
            acc = model.score(val_data, val_y.codes)  # type: ignore[attr-defined]
            print(f"Val acc: {acc:.4f},", end=" ")
            if log_file:
                log.write(f"Val acc: {acc:.4f}, ")
            end_time = time.time()
            print(f"Time: {(end_time - start_time) / 60:.2f} mins", end=" ")
            print()
            if log_file:
                log.write(f"Time: {(end_time - start_time) / 60:.2f} mins\n")
                log.flush()
        if log_file:
            log.close()



    def predict(self, X):
        overall_preds, _ = self.predict_by_fold(X)
        return overall_preds

    def predict_by_fold(self, X):
        """Return ensemble predictions alongside per-fold model predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features indexed by sample identifier. This frame should include all
            expression columns; each fold model will subset to its selected genes.

        Returns
        -------
        overall : pandas.Series
            Predictions from the ensemble obtained by averaging per-fold probabilities.
        per_fold : dict[int, pandas.Series]
            Mapping from fold identifier to that fold model's predictions on ``X``.
        """
        if not self.models:
            raise ValueError("No trained models are available for prediction.")

        fold_ids = sorted(self.models.keys())
        probabilities = []
        per_fold_predictions = {}

        for fold in fold_ids:
            genes = self.genes_used_by_model.get(fold)
            if genes is None:
                raise ValueError(
                    f"Gene set for fold {fold} is not available. Train or load a model with stored gene subsets."
                )
            pred_X = X.loc[:, genes]
            model = self.models[fold]
            fold_proba = np.asarray(model.predict_proba(pred_X))
            probabilities.append(fold_proba)
            fold_codes = np.argmax(fold_proba, axis=1).astype(int)
            per_fold_predictions[fold] = pd.Series(
                pd.Categorical.from_codes(fold_codes, categories=self.classes),
                index=X.index
            )

        avg_proba = np.mean(probabilities, axis=0)
        avg_codes = np.argmax(avg_proba, axis=1).astype(int)
        overall_predictions = pd.Series(
            pd.Categorical.from_codes(avg_codes, categories=self.classes),
            index=X.index
        )

        return overall_predictions, per_fold_predictions

    def accuracy(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()