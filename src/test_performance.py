"""Evaluate LightGBM model performance on a specified fold.

This script loads a saved `LGBM_Model`, gathers predictions for a target
fold (defaulting to the held-out `TEST_FOLD`), computes summary metrics,
and generates plots including per-lineage accuracies and optional confusion
matrices for user-specified lineages.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Categorical, CategoricalDtype
from sklearn.metrics import confusion_matrix

from model import TEST_FOLD, LGBM_Model


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a trained LightGBM ensemble on a specific fold."
	)
	parser.add_argument(
		"--model_path",
		"-m",
		default="./models/lgbm_model.pkl",
		help="Path to the pickled LightGBM model ensemble.",
	)
	parser.add_argument(
		"--data_dir",
		"-d",
		default=None,
		help="Optional override for the directory containing fold CSV files.",
	)
	parser.add_argument(
		"--fold",
		"-f",
		type=int,
		default=TEST_FOLD,
		help="Fold number to evaluate. Defaults to the model's TEST_FOLD.",
	)
	parser.add_argument(
		"--output_dir",
		"-o",
		default="./reports/performance/",
		help="Directory where metrics, tables, and plots will be written.",
	)
	parser.add_argument(
		"--confusion_lineages",
		"-c",
		type=str,
		default="all",
		help=(
			"Comma-separated list of lineage names for which to produce a"
			" confusion matrix plot."
		),
	)
	parser.add_argument(
		"--save_predictions",
		type=str,
		default="predictions.csv",
		help="Filename (inside the output directory) for per-sample predictions.",
	)
	return parser.parse_args()


def _ensure_output_dir(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)


def _load_model(model_path: str, data_dir: Optional[str]) -> LGBM_Model:
	model = LGBM_Model.load_model(model_path)
	if data_dir:
		model.data_dir = data_dir
	return model


def _get_fold_predictions(
	model: LGBM_Model, fold: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[int, pd.Series]]:
	X_raw, y_raw = model.get_fold_data(fold, subset_to_model=False)
	X = cast(pd.DataFrame, X_raw)

	if isinstance(y_raw, pd.Series):
		y_series = y_raw
	else:
		y_categorical = cast(Categorical, y_raw)
		y_series = pd.Series(y_categorical, index=X.index, name="true_lineage")

	overall_preds, per_fold_preds = model.predict_by_fold(X)

	return X, y_series, overall_preds, per_fold_preds


def _compute_accuracy_by_lineage(
	y_true: pd.Series, y_pred: pd.Series, categories: Iterable[str]
) -> pd.DataFrame:
	records = []
	for lineage in categories:
		mask = y_true == lineage
		support = int(mask.sum())
		if support == 0:
			continue
		accuracy = float((y_pred[mask] == y_true[mask]).mean())
		records.append({
			"Lineage": str(lineage),
			"accuracy": accuracy,
			"support": support,
		})
	return pd.DataFrame.from_records(records)


def _infer_categories(y_true: pd.Series, fallback: Iterable[str]) -> List[str]:
	if isinstance(getattr(y_true, "dtype", None), CategoricalDtype):
		return [str(cat) for cat in y_true.cat.categories]
	# fallback to observed labels in data preserving sorted order
	observed = pd.unique(y_true.astype(str))
	if len(observed) > 0:
		return observed.tolist()
	return [str(cat) for cat in fallback]




def _plot_accuracy_summary(
	mean_accuracy_by_lineage: pd.DataFrame,
	per_fold_accuracy: pd.DataFrame,
	overall_accuracy: float,
	lineage_plot_path: Path,
	per_fold_overall_accuracy: Dict[str, float],
	fold_plot_path: Path,
) -> None:
	if mean_accuracy_by_lineage.empty:
		return

	ordered = mean_accuracy_by_lineage.sort_values("mean_fold_accuracy").reset_index(drop=True)
	labels_with_support = ordered.apply(
		lambda row: f"{row['Lineage']} ({int(row['support'])})", axis=1
	)

	# Lineage accuracy distribution box plot
	box_data = []
	for lineage in ordered["Lineage"]:
		lineage_data = per_fold_accuracy[per_fold_accuracy["Lineage"] == lineage]
		if lineage_data.empty:
			box_data.append([])
		else:
			box_data.append(lineage_data["accuracy"].to_numpy())

	fig_box, ax_box = plt.subplots(figsize=(10, max(4, len(ordered) * 0.5)))
	boxplot = ax_box.boxplot(
		box_data,
		vert=False,
		tick_labels=list(labels_with_support),
		patch_artist=True,
		boxprops=dict(facecolor="#4C72B0", alpha=0.4),
		medianprops=dict(color="#4C72B0", linewidth=2),
		whiskerprops=dict(color="#2A4E6C"),
		capprops=dict(color="#2A4E6C"),
		flierprops=dict(marker="o", markerfacecolor="#4C72B0", markersize=4, alpha=0.6),
	)

	mean_points = ordered["mean_fold_accuracy"].to_numpy()
	y_positions = np.arange(1, len(ordered) + 1)
	ax_box.scatter(
		mean_points,
		y_positions,
		s=55,
		color="#1B2838",
		marker="D",
		label="Mean across folds",
		zorder=4,
	)

	for bp in boxplot["boxes"]:
		bp.set_edgecolor("#2A4E6C")
		bp.set_linewidth(1.2)

	ax_box.axvline(
		overall_accuracy,
		color="#DD8452",
		linestyle="--",
		linewidth=2,
		label=f"Ensemble overall: {overall_accuracy:.3f}",
	)

	ax_box.set_xlabel("Accuracy")
	ax_box.set_title("Lineage accuracy distribution across folds")
	ax_box.set_xlim(0, 1)
	ax_box.set_ylabel("Lineage")
	ax_box.legend(loc="lower right", fontsize="small", frameon=True)
	plt.tight_layout()
	fig_box.savefig(lineage_plot_path, bbox_inches="tight")
	plt.close(fig_box)

	# Overall accuracy by fold bar chart
	if per_fold_overall_accuracy:
		sorted_folds = sorted(per_fold_overall_accuracy.items(), key=lambda item: int(item[0]))
		fold_labels = [f"Fold {fold}" for fold, _ in sorted_folds]
		fold_values = [value for _, value in sorted_folds]

		fig_bar, ax_bar = plt.subplots(figsize=(max(6, len(sorted_folds) * 1.2), 4))
		ax_bar.bar(
			np.arange(len(sorted_folds)),
			fold_values,
			color="#55A868",
			alpha=0.75,
		)
		ax_bar.axhline(
			overall_accuracy,
			color="#DD8452",
			linestyle="--",
			linewidth=2,
		    label=f"Ensemble overall: {overall_accuracy:.3f}",
		)
		ax_bar.set_xticks(np.arange(len(sorted_folds)))
		ax_bar.set_xticklabels(fold_labels, rotation=45, ha="right")
		ax_bar.set_ylim(0, 1)
		ax_bar.set_ylabel("Accuracy")
		ax_bar.set_title("Overall accuracy by fold")
		ax_bar.legend(loc="upper right", fontsize="small", frameon=True)
		plt.tight_layout()
		fig_bar.savefig(fold_plot_path, bbox_inches="tight")
		plt.close(fig_bar)


def _prepare_confusion_lineages(arg: Optional[str]) -> Optional[List[str]]:
	if arg is None:
		return None
	tokens = [token.strip() for token in arg.split(",") if token.strip()]
	if not tokens:
		return None
	if any(token.lower() == "all" for token in tokens):
		return []
	return tokens


def _plot_confusion_matrix(
	y_true: pd.Series,
	y_pred: pd.Series,
	labels: List[str],
	output_path: Path,
) -> None:
	mask = y_true.isin(labels)
	if mask.sum() == 0:
		raise ValueError(
			"None of the specified lineages were present in the provided fold."
		)
	y_true_subset = y_true[mask]
	y_pred_subset = y_pred[mask]
	# Align predictions with filtered indices
	y_pred_subset = y_pred_subset.reindex(y_true_subset.index)

	matrix = confusion_matrix(
		y_true_subset, y_pred_subset, labels=labels, normalize=None
	)
	with np.errstate(divide="ignore", invalid="ignore"):
		row_sums = matrix.sum(axis=1, keepdims=True)
		normalized = np.divide(matrix, row_sums, where=row_sums != 0)

	fig, ax = plt.subplots(figsize=(1.5 * len(labels), 1.2 * len(labels)))
	cmap = plt.get_cmap("Blues")
	im = ax.imshow(normalized, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
	ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.set_yticklabels(labels)
	ax.set_xlabel("Predicted lineage")
	ax.set_ylabel("True lineage")
	ax.set_title("Confusion matrix (count with row-normalized color)")

	for i in range(len(labels)):
		for j in range(len(labels)):
			count = matrix[i, j]
			frac = normalized[i, j] if row_sums[i, 0] else 0.0
			ax.text(
				j,
				i,
				f"{count}\n({frac:.2f})",
				ha="center",
				va="center",
				color="black" if frac < 0.6 else "white",
				fontsize=9,
			)

	plt.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = _parse_args()
	output_dir = Path(args.output_dir)
	_ensure_output_dir(output_dir)

	model = _load_model(args.model_path, args.data_dir)

	if args.fold not in set(range(model.num_folds)):
		raise ValueError(
			f"Requested fold {args.fold} is outside the range 0..{model.num_folds - 1}."
		)

	X, y_true, ensemble_pred, per_fold_preds = _get_fold_predictions(model, args.fold)

	categories = _infer_categories(y_true, model.classes)
	y_true_str = y_true.astype(str)
	ensemble_pred_str = ensemble_pred.astype(str)
	per_fold_pred_str = {fold: preds.astype(str) for fold, preds in per_fold_preds.items()}

	overall_accuracy = float((ensemble_pred_str == y_true_str).mean())
	ensemble_accuracy = _compute_accuracy_by_lineage(y_true_str, ensemble_pred_str, categories)
	ensemble_accuracy = ensemble_accuracy.rename(columns={"accuracy": "ensemble_accuracy"})

	per_fold_accuracy_frames: List[pd.DataFrame] = []
	per_fold_overall_accuracy: Dict[str, float] = {}
	for fold, preds_str in per_fold_pred_str.items():
		per_fold_overall_accuracy[str(fold)] = float((preds_str == y_true_str).mean())
		fold_accuracy = _compute_accuracy_by_lineage(y_true_str, preds_str, categories)
		if fold_accuracy.empty:
			continue
		fold_accuracy = fold_accuracy.assign(fold=fold)
		per_fold_accuracy_frames.append(fold_accuracy)

	if per_fold_accuracy_frames:
		per_fold_accuracy = pd.concat(per_fold_accuracy_frames, ignore_index=True)
	else:
		per_fold_accuracy = pd.DataFrame(columns=["Lineage", "accuracy", "support", "fold"])

	if not per_fold_accuracy.empty:
		per_fold_accuracy = per_fold_accuracy.sort_values(["Lineage", "fold"]).reset_index(drop=True)
		fold_summary = (
			per_fold_accuracy
			.groupby("Lineage", as_index=False)
			.agg(
				mean_fold_accuracy=("accuracy", "mean"),
				support=("support", "first"),
				std_fold_accuracy=("accuracy", "std"),
				fold_count=("fold", "nunique"),
			)
		)
		fold_summary["std_fold_accuracy"] = fold_summary["std_fold_accuracy"].fillna(0.0)
	else:
		fold_summary = ensemble_accuracy.rename(columns={"ensemble_accuracy": "mean_fold_accuracy"})
		fold_summary["std_fold_accuracy"] = 0.0
		fold_summary["fold_count"] = 0

	accuracy_summary = fold_summary.merge(
		ensemble_accuracy[["Lineage", "ensemble_accuracy"]],
		on="Lineage",
		how="left",
	)
	accuracy_summary = accuracy_summary[
		["Lineage", "support", "mean_fold_accuracy", "std_fold_accuracy", "fold_count", "ensemble_accuracy"]
	]

	metrics_summary = {
		"fold": args.fold,
		"num_samples": int(len(y_true)),
		"overall_accuracy": overall_accuracy,
		"per_fold_overall_accuracy": per_fold_overall_accuracy,
		"lineages_evaluated": list(categories),
	}

	accuracy_path = output_dir / "accuracy_by_lineage.csv"
	accuracy_summary.to_csv(accuracy_path, index=False)

	per_fold_accuracy_path = output_dir / "accuracy_by_lineage_per_fold.csv"
	per_fold_accuracy.to_csv(per_fold_accuracy_path, index=False)

	predictions_path = output_dir / args.save_predictions
	predictions_df = pd.DataFrame(
		{
			"sample_id": X.index,
			"true_lineage": y_true_str,
			"predicted_lineage": ensemble_pred_str,
		}
	)
	for fold in sorted(per_fold_pred_str.keys()):
		predictions_df[f"predicted_fold_{fold}"] = per_fold_pred_str[fold].reindex(predictions_df.index).values
	predictions_df.to_csv(predictions_path, index=False)

	metrics_path = output_dir / "metrics.json"
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(metrics_summary, f, indent=2)

	accuracy_box_plot_path = output_dir / "accuracy_by_lineage.png"
	fold_accuracy_plot_path = output_dir / "accuracy_by_fold.png"
	_plot_accuracy_summary(
		accuracy_summary,
		per_fold_accuracy,
		overall_accuracy,
		accuracy_box_plot_path,
		per_fold_overall_accuracy,
		fold_accuracy_plot_path,
	)

	lineages_for_confusion = _prepare_confusion_lineages(args.confusion_lineages)
	if lineages_for_confusion is not None:
		lineage_list: List[str] = list(categories) if len(lineages_for_confusion) == 0 else lineages_for_confusion
		missing = [lin for lin in lineage_list if lin not in categories]
		if missing:
			raise ValueError(
				"The following requested lineages are not present in the evaluation "
				f"data: {missing}"
			)
		conf_path = output_dir / (
			"confusion_matrix.png"
		)
		_plot_confusion_matrix(y_true_str, ensemble_pred_str, lineage_list, conf_path)

	print(f"Evaluation complete. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
	main()
