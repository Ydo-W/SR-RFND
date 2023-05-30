# SR-RFND：Symbolic regression for redundant features and noise data

## Description
This is the implementation of manuscript “***”. 

## System requirements
#### Installation
* Python 	3.7+
* PtTorch 1.8.0+cu111
* PySR 0.12.0

## Quick start
### Examples
We provide in this repository the verification of SR-RFND on two known formulas, as follows:
1. $$E=\frac{q_1}{4\pi\epsilon r^2},$$ 
2. $$E=\frac{1}{2}m(v^2 + u^2 + w^2)$$

The demos for these two examples are stored under `./examples_of_known_formulas-1/` and `examples_of_known_formulas-2` respectively. 
### Data
Users can use our datasets, stored under `./{example folder}/datasets/`, or re-generate them by running `./{example folder}/dataset_made.py`.
### Running
#### Direct Symbolic Regression
1. Run `./{example folder}/directly_SR/SR.py`.
#### Feature Filtering & Symbolic Regression
1. Run `./{example folder}/pipeline_on_noise_data/train_baseline.py` to train the baseline network for performing multivariate regression task. 
2. Run `./{example folder}/pipeline_on_noise_data/RFE_feature_selection.py` to perform the feature filtering, the results will be reported in `Feature importance.log` under the same folder. 
3. Run `./{example folder}/pipeline_on_noise_data/SR.py` to perform symbolic regression and evaluation.
#### Sample Filtering & Feature Filtering & Symbolic Regression
1. Run `./{example folder}/train_baseline_ncr/train_baseline.py` to train the baseline network with neighborhood consistency regularization. Then move the trained model file to `./{example folder}/pipeline_on_selected_data/checkpoints/`.
2. Run `./{example folder}/pipeline_on_selected_data/data_selection.py` to perform sample filtering.
3. Run `./{example folder}/pipeline_on_noise_data/train_baseline.py` to train the baseline network for performing multivariate regression task. 
4. Run `./{example folder}/pipeline_on_noise_data/RFE_feature_selection.py` to perform the feature filtering, the results will be reported in `Feature importance.log` under the same folder. 
5. Run `./{example folder}/pipeline_on_noise_data/SR.py` to perform symbolic regression and evaluation.





