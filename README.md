# Biostats Final Project

## Env
- create conda env with `conda env create --prefix ./env -f environment.yaml`
- activate with `conda activate ./env`

## Data
 - Download files `Model.csv, OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv, OmicsProfiles.csv` from `https://depmap.org/portal/data_page/?tab=allData` and place in `./data/raw`
 - create folds with below
 - will create 5 folds
 ```
 python .\src\create_folds.py -X .\data\raw\OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv -y .\data\raw\OmicsProfiles.csv
 ```

## Train
 - model details can be found in `./src/model.py`
 - Train with `python ./src/train.py`

 ## Evaluation
  - calculate test accuracies and create plots with `python ./src/test_performance.py`
  - calculate pearson correlation between performance metrics with `python ./src/calc_correlations.py`
  - calculate shap values and create feature importance plot with `python ./src/shap_reports.py`