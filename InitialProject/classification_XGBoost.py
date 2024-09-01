import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from optuna import Trial, create_study
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


class XGBObjective:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, trial: Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 20),
        }
        bst = XGBClassifier(**params)
        bst.fit(self.X_train, self.y_train)
        bst_pred_proba = bst.predict_proba(self.X_test)[:, 1]
        score = roc_auc_score(self.y_test, bst_pred_proba)
        return score


train_path = "AppliedML2024/InitialProject/Data files/AppML_InitialProject_train.h5"
test_path = "AppliedML2024/InitialProject/Data files/AppML_InitialProject_test_classification.h5"
train_df = pd.read_hdf(train_path)
test_df = pd.read_hdf(test_path)
data_train = train_df.iloc[:, :-2]
data_cls_target = train_df.iloc[:, -2]

# X_train, X_test, y_train, y_test = train_test_split(
#     data_train, data_cls_target, test_size=0.5, random_state=42
# )
# objective = XGBObjective(X_train, X_test, y_train, y_test)
# study = create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
# best_trial = study.best_trial
# optimal_params = best_trial.params
# bst = XGBClassifier(**optimal_params)
# bst.fit(X_train, y_train)
# explainer = shap.TreeExplainer(bst)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=data_train.columns), plot_type="bar")
# plt.show()
optimal_params_all_features = {
    "n_estimators": 450,
    "learning_rate": 0.10418107567836463,
    "max_depth": 9,
    "lambda": 0.00028831683796263027,
    "alpha": 0.0783639051656269,
    "subsample": 0.9733959948302068,
    "colsample_bytree": 0.9323041807128275,
    "gamma": 0.0015913911062558628,
}

top_features = [
    "p_sigmad0",
    "p_TRTPID",
    "pX_MultiLepton",
    "p_numberOfInnermostPixelHits",
    "p_dPOverP",
    "p_ptPU30",
    "p_deltaEta1",
    "pX_deltaEta1",
    "p_Rhad",
    "p_d0",
    "p_numberOfPixelHits",
    "p_etcone20",
    "p_deltaPhiRescaled2",
    "p_pt_track",
    "p_Reta",
    "p_weta2",
    "pX_deltaPhiFromLastMeasurement",
    "pX_ptconecoreTrackPtrCorrection",
    "p_f1",
    "p_charge",
]

X_train, X_test, y_train, y_test = train_test_split(
    data_train[top_features], data_cls_target, test_size=0.5, random_state=42
)
# objective = XGBObjective(X_train, X_test, y_train, y_test)
# study = create_study(direction="maximize")
# study.optimize(objective, n_trials=200)
# best_trial = study.best_trial
# optimal_params = best_trial.params
optimal_params_top20 = {
    "n_estimators": 926,
    "learning_rate": 0.019398536536408783,
    "max_depth": 6,
    "lambda": 0.0007847223067083008,
    "alpha": 3.044193956259971e-08,
    "subsample": 0.6494191061307464,
    "colsample_bytree": 0.8638996606090434,
    "gamma": 0.31864519686441345,
}

bst = XGBClassifier(**optimal_params_top20)
bst.fit(X_train, y_train)
pred_proba_validation = bst.predict_proba(X_test)[:, 1]
# Value: 0.07327356519391241
log_loss_validation = log_loss(y_test, pred_proba_validation)
pred_proba_final = bst.predict_proba(test_df[top_features])[:, 1]

csv_out_path = (
    "AppliedML2024/InitialProject/Solution_files/Classification_AliAhmad_XGBoost.csv"
)
features_out_path = "AppliedML2024/InitialProject/Solution_files/Classification_AliAhmad_VariableList.csv"

# with open(features_out_path, "w") as f:
#     for feature in top_features:
#         f.write(f"{feature},\n")
# pd.Series(pred_proba_final).to_csv(csv_out_path, sep=",", header=None)
