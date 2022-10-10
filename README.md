# MLflow for begginers

## In this tutorial, you'll learn how to manage a Machine Learning project lifecycle with MLflow, locally

### Install MLflow on linux environement

    pip install mlflow[extras]

### Explore the "titanic_classification.ipynb" notebook
   This is a fairly simple classification example. 
   It illustrates how to apply different preprocessing and feature extraction pipelines.
   
   For more details check the link below :
   
    https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#column-transformer-with-mixed-types
    
### Integrate MLflow to the pipline : 
   In order to track experiments, register models and deploy them you will have to integrate some mlflow functionalities in your code, as shown in the "titanic_classification_mlflow.ipynb" notebook.
  
   #### Import MLflow
        import mlflow
   #### Set a tracking server on localhost
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
   #### Create a new experiment : 
        mlflow.create_experiment(EXPERIMENT_NAME)
   #### Set an experimment or create it if doesn't exist
        mlflow.set_experiment(EXPERIMENT_NAME)
   #### Log runs with mlflow
   Once you start a Run youu can log parameters, metrics and the model with its artifacts.
   Runs will be stored in mlruns directory wherever your run the code.
   
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        
        # mlflow log parameters
        mlflow.log_param("classifier_C",param)
        
        # mlflow log metrics
        mlflow.log_metric("pipeline_test_score",accuracy_score)
        
        # mlflow log model
        mlflow.sklearn.log_model(clf, "classifier_titanic")

### Launch MLflow on localhost with a Tracking Server:
    Terminal:§ mlflow ui --backend-store-uri sqlite:///mlruns.db
   
   MLflow UI will appear on your browser, so you can :
   - Track experiments
   - Compare runs 
   - Register models
   - Manage transitions 
   
   You can also create regisries, and manipluate them by code : https://www.mlflow.org/docs/latest/model-registry.html
  
