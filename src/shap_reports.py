import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model import LGBM_Model

RANDOM_SEED = 4872224


def _parse_folds(fold_arg: str, available_folds):
	if fold_arg is None or fold_arg.strip().lower() == "all":
		return sorted(available_folds)
	folds = []
	for token in fold_arg.split(","):
		token = token.strip()
		if not token:
			continue
		try:
			fold = int(token)
		except ValueError as exc:
			raise ValueError(f"Fold '{token}' is not an integer.") from exc
		if fold not in available_folds:
			raise ValueError(f"Fold {fold} is not available. Choose from {sorted(available_folds)}.")
		folds.append(fold)
	if not folds:
		raise ValueError("No valid folds were provided.")
	return folds


def _sample_frame(X: pd.DataFrame, y: pd.Categorical, sample_rows: int, random_state: int):
	categories = getattr(y, "categories", None)
	if isinstance(y, pd.Categorical):
		y_series = pd.Series(y, index=X.index)
	elif isinstance(y, pd.Series):
		y_series = y
	else:
		y_series = pd.Series(y, index=X.index)
	if sample_rows and len(X) > sample_rows:
		sampled_index = X.sample(n=sample_rows, random_state=random_state).index
		X_sampled = X.loc[sampled_index]
		y_sampled = y_series.loc[sampled_index]
		if categories is not None:
			y_sampled = pd.Categorical(y_sampled, categories=categories)
		return X_sampled, y_sampled
	return X, y


def _as_stack(shap_values, num_features: int):
	if isinstance(shap_values, list):
		return np.stack([np.asarray(v) for v in shap_values], axis=0)

	values = np.asarray(shap_values.values) if hasattr(shap_values, "values") else np.asarray(shap_values)

	if values.ndim == 1:
		if values.shape[0] != num_features:
			raise ValueError(f"Unexpected SHAP value shape {values.shape}; expected {num_features} features.")
		return values[np.newaxis, np.newaxis, :]

	if values.ndim == 2:
		if values.shape[1] == num_features:
			return values[np.newaxis, :, :]
		if values.shape[0] == num_features:
			return values[np.newaxis, :, :].swapaxes(1, 2)
		raise ValueError(f"Cannot align SHAP array of shape {values.shape} with {num_features} features.")

	if values.ndim == 3:
		if values.shape[1] == num_features:
			# (samples, features, outputs)
			return np.moveaxis(values, -1, 0)
		if values.shape[2] == num_features:
			# (samples, outputs, features)
			return np.moveaxis(values, 1, 0)
		if values.shape[0] == num_features:
			# (features, samples, outputs)
			return np.moveaxis(values, -1, 0)
		raise ValueError(f"Cannot align SHAP array of shape {values.shape} with {num_features} features.")

	if values.ndim == 4 and values.shape[-1] == num_features:
		# (batch, samples, outputs, features)
		squeezed = values.reshape(-1, values.shape[-2], values.shape[-1])
		return np.moveaxis(squeezed, 1, 0)

	raise ValueError(f"Unsupported SHAP values shape {values.shape}")


def _prepare_expected_values(expected_value, class_names):
	if isinstance(expected_value, list):
		return {str(class_names[i]) if i < len(class_names) else str(i): float(val) for i, val in enumerate(expected_value)}
	return {str(class_names[0] if class_names else 0): float(expected_value)}


def _save_beeswarm(explanation, output_path: Path, beeswarm_max_display: int):
	shap.plots.beeswarm(explanation, show=False, max_display=beeswarm_max_display)
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()


def _save_feature_bar(mean_abs_values, feature_names, output_path: Path, bar_max_display: int):
	order = np.argsort(mean_abs_values)[::-1][:bar_max_display]
	ordered_features = [str(f) for f in np.asarray(feature_names)[order].tolist()]
	ordered_values = np.asarray(mean_abs_values)[order].tolist()
	plt.figure(figsize=(8, 6))
	plt.barh(ordered_features[::-1], ordered_values[::-1])
	plt.xlabel("Mean |SHAP value|")
	plt.ylabel("Feature")
	plt.title("Top features by mean absolute SHAP value")
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()


def _write_json(data, output_path: Path):
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)


def generate_fold_reports(model: LGBM_Model, fold: int, output_dir: Path, sample_rows: int, random_state: int, beeswarm_max_display: int, bar_max_display: int):
	X, y = model.get_train_data(model.train_val_folds - {fold})
	X_sampled, y_sampled = _sample_frame(X, y, sample_rows, random_state)

	explainer = shap.TreeExplainer(model.models[fold], data=X_sampled, model_output="probability")
	explanation = explainer(X_sampled)
	shap_values_raw = explanation.values
	shap_stack = _as_stack(shap_values_raw, X_sampled.shape[1])
	# expected_values = _prepare_expected_values(explainer.expected_value, model.classes)
	expected_values = None

	mean_abs_values = np.mean(np.abs(shap_stack), axis=(0, 1))
	feature_names = X_sampled.columns.to_numpy()

	fold_dir = output_dir / f"fold_{fold}"
	fold_dir.mkdir(parents=True, exist_ok=True)

	# _save_beeswarm(explanation, fold_dir / "shap_summary_beeswarm.png", beeswarm_max_display)
	class_idx = 0  # or whichever class you care about
	_save_beeswarm(explanation[..., class_idx], fold_dir / "shap_summary_beeswarm.png", beeswarm_max_display)
	_save_feature_bar(mean_abs_values, feature_names, fold_dir / "shap_feature_importance.png", bar_max_display)

	feature_df = pd.DataFrame({
		"feature": feature_names,
		"mean_abs_shap": mean_abs_values,
		"fold": fold
	}).sort_values("mean_abs_shap", ascending=False)
	feature_df.to_csv(fold_dir / "feature_importance.csv", index=False)

	meta = {
		"fold": fold,
		"num_samples": int(X_sampled.shape[0]),
		"num_features": int(X_sampled.shape[1]),
		"expected_values": expected_values,
		"top_features": feature_df.head(bar_max_display).to_dict(orient="records")
	}
	_write_json(meta, fold_dir / "summary.json")

	shap_tensor = pd.DataFrame(shap_stack.mean(axis=0), columns=feature_names, index=X_sampled.index)

	return {
		"feature_importance": feature_df,
		"shap_tensor": shap_tensor,
		"X": X_sampled,
		"y": y_sampled
	}


def aggregate_importance(output_dir: Path, bar_max_display: int):
	fold_importances = []
	for fold_dir in output_dir.glob("fold_*"):
		fi_path = fold_dir / "feature_importance.csv"
		if fi_path.exists():
			df = pd.read_csv(fi_path)
			fold_importances.append(df)
	print(f"Aggregating feature importance from {len(fold_importances)} folds")
	combined = pd.concat([frame for frame in fold_importances], ignore_index=True)
	agg = (
		combined
		.groupby("feature")
		["mean_abs_shap"]
		.agg(["mean", "std", "max", "sum", "count"])
		.rename(columns={
			"mean": "mean_abs_shap",
			"std": "std_abs_shap",
			"max": "max_abs_shap",
			"sum": "sum_abs_shap",
			"count": "num_folds_present"
		})
		.sort_values("mean_abs_shap", ascending=False)
	)
	agg.to_csv(output_dir / "aggregate_feature_importance.csv")

	top_features = agg.head(bar_max_display)
	plt.figure(figsize=(8, 6))
	plt.barh(top_features.index[::-1], top_features["mean_abs_shap"][::-1])
	plt.xlabel("Mean |SHAP value|")
	plt.ylabel("Feature")
	plt.title("Global top features across folds")
	plt.tight_layout()
	plt.savefig(output_dir / "aggregate_feature_importance.png", bbox_inches="tight")
	plt.close()


def parse_args():
	parser = argparse.ArgumentParser(description="Generate SHAP plots and feature importance summaries for trained LightGBM models.")
	parser.add_argument("--model_path", "-m", type=str, default="./models/lgbm_model.pkl", help="Path to the pickled LightGBM model ensemble.")
	parser.add_argument("--data_dir", "-d", type=str, default=None, help="Optional override for the data directory containing fold CSVs.")
	parser.add_argument("--output_dir", "-o", type=str, default="./reports/shap/", help="Directory to write plots and tables.")
	parser.add_argument("--folds", type=str, default="all", help="Comma-separated list of folds to analyse or 'all'.")
	parser.add_argument("--sample_rows", type=int, default=1000, help="Maximum number of rows per fold to use for SHAP (set to 0 for all rows).")
	parser.add_argument("--max_display", type=int, default=25, help="Default maximum number of features to display when specific overrides aren't provided.")
	parser.add_argument("--beeswarm_max_display", type=int, default=None, help="Maximum number of features for beeswarm summary plots. Defaults to --max_display.")
	parser.add_argument("--bar_max_display", type=int, default=None, help="Maximum number of features for bar charts and tabular summaries. Defaults to --max_display.")
	parser.add_argument("--random_state", type=int, default=RANDOM_SEED, help="Random seed for row subsampling.")
	return parser.parse_args()


def main():
	args = parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	model = LGBM_Model.load_model(args.model_path)
	if args.data_dir:
		model.data_dir = args.data_dir

	available_folds = set(model.models.keys())
	folds = _parse_folds(args.folds, available_folds)

	beeswarm_max_display = args.beeswarm_max_display if args.beeswarm_max_display is not None else args.max_display
	bar_max_display = args.bar_max_display if args.bar_max_display is not None else args.max_display

	shap_tensors = []

	for fold in folds:
		# if feature_importance.csv already exists for this fold, skip
		fold_dir = output_dir / f"fold_{fold}"
		if (fold_dir / "feature_importance.csv").exists():
			print(f"Skipping fold {fold} as reports already exist.")
			continue
		print(f"Generating SHAP reports for fold {fold},", end="", flush=True)
		start_time = time.time()
		report = generate_fold_reports(
			model=model,
			fold=fold,
			output_dir=output_dir,
			sample_rows=args.sample_rows,
			random_state=args.random_state,
			beeswarm_max_display=beeswarm_max_display,
			bar_max_display=bar_max_display
		)
		shap_tensors.append(report["shap_tensor"])
		elapsed_mins = (time.time() - start_time) / 60.0
		print(f" done in {elapsed_mins:.2f} mins.")


	aggregate_importance(output_dir, bar_max_display)

	if shap_tensors:
		combined_tensor = pd.concat(shap_tensors, axis=0)
		combined_tensor.to_csv(output_dir / "mean_shap_values_by_sample.csv")

	print(f"SHAP reports written to {output_dir.resolve()}")


if __name__ == "__main__":
	main()

