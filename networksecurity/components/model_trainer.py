import os, sys
from networksecurity.logging.logger import logging 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import save_numpy_array_data, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
import mlflow
import mlflow.sklearn
# import dagshub
# dagshub.init(repo_owner='ArpitKadam', repo_name='Network_Security', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:      
            logging.error(f"Error in ModelTrainer: {str(e)}")
            raise NetworkSecurityException(f"Error in ModelTrainer: {str(e)}", sys)

    def track_mlflow(self, best_model, classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            accuracy_score = classificationmetric.accuracy_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("accuracy_score", accuracy_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="NetworkSecurityModel")

    def train_model(self, x_train, y_train, x_test, y_test) -> NetworkModel:
        try:
            logging.info("Choosing Model and Training the Model")
            models = {
                "Logistic Regression": LogisticRegression(verbose=1, n_jobs=-1),
                "Random Forest": RandomForestClassifier(verbose=1, n_jobs=-1),
                "Ada Boost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Support Vector Machine": SVC(),
                "Extra Tree": ExtraTreeClassifier()
            }

            logging.info("Hyperparameter Tuning")
            params = {
                "Logistic Regression": {
                    "penalty": ["l2"], 
                    "C": [1.0, 0.5],
                    "solver": ["lbfgs", "saga"], 
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", None],
                },
                "Ada Boost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [1.0, 0.1, 0.01],
                },
                "Bagging": {
                    "n_estimators": [10, 50],
                    "max_samples": [0.5, 1.0],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", None],
                },
                "Support Vector Machine": {
                    "C": [1.0, 0.5],
                    "kernel": ["rbf", "sigmoid"],
                    "degree": [3, 5],
                    "gamma": ["auto"],
                },
                "Extra Tree": {
                    "criterion": ["gini"],
                    "splitter": ["best", "random"],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", None],
                },
            }

            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models, params=params
            )
            logging.info("Hyperparameter Tuning Completed")
            logging.info(f"Model Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best Model Score: {best_model_score}")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"Best Model Name: {best_model_name}")
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            logging.info(f"Classification Train Metric: {classification_train_metric}")  

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Classification Test Metric: {classification_test_metric}")

            ## Track with MLFlow - DISABLED
            # logging.info("Tracking Trained Model with MLFlow")
            # self.track_mlflow(best_model, classification_train_metric)
            # logging.info("Tracking Test Model with MLFlow")
            # self.track_mlflow(best_model, classification_test_metric)
            # logging.info("Completed Tracking Model with MLFlow")

            processor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            make_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(make_dir_path, exist_ok=True)

            logging.info(f"Saving the Trained Model: {self.model_trainer_config.trained_model_file_path}")
            save_object("final_models/model.pkl", best_model)

            Network_Model = NetworkModel(processor=processor, model=best_model)
            logging.info(f"Network Model: {Network_Model}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise NetworkSecurityException(f"Error in train_model: {str(e)}", sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact: 
        try:
            logging.info("Initiating Model Trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            logging.info(f"Train File Path: {train_file_path}")
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            logging.info(f"Test File Path: {test_file_path}")

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test
            )
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise NetworkSecurityException(f"Error in initiate_model_trainer: {str(e)}", sys)