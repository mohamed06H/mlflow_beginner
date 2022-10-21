# MLflow for beginners

## In this tutorial, you'll learn how to manage a Machine Learning project lifecycle with MLflow, locally

### Install MLflow on linux environment

    pip install mlflow[extras]

**_NOTE:_**  Best Practice for windows users : Work on virtual environments (pyenv or conda) within WSL2

https://github.com/pyenv/pyenv#basic-github-checkout

### Explore the "titanic_classification.ipynb" notebook
   This is a fairly simple classification example. 
   It illustrates how to apply different preprocessing and feature extraction pipelines.
   
   For more details check the link below :
   
    https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#column-transformer-with-mixed-types
    
### Integrate MLflow to the pipeline : 
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

### Launch MLflow UI on localhost with a Tracking Server:
    Terminal:§ mlflow ui --backend-store-uri sqlite:///mlruns.db
   
   MLflow UI will appear on your browser, so you can :
   - Track experiments
   <img width="959" alt="tracking " src="https://user-images.githubusercontent.com/114097516/197213970-b8bbf088-54af-449e-a15f-e101de546da1.png">
   
   - Track runs (informations, parameters, metrics, artifacts) 
   
   In artifacts of each run, you'll find MLmodel, conda.yaml and other files which discribes the virtual environment required to reproduce the model.
   
   ![Capture web_21-10-2022_153949_127 0 0 1](https://user-images.githubusercontent.com/114097516/197209863-517da0d5-17fb-481a-809a-ec73a334de83.jpeg)
   
   - Compare runs
   <img width="960" alt="compare2" src="https://user-images.githubusercontent.com/114097516/197214582-0c5bfc66-3bb4-4697-94db-17795167c7f7.png">

   - Register models
   <img width="960" alt="register" src="https://user-images.githubusercontent.com/114097516/197211487-571abb6e-7b7b-4ccb-adb3-86b1b1da3838.png">

   - Manage versions and transitions 
   <img width="960" alt="staging" src="https://user-images.githubusercontent.com/114097516/197212265-91ccf8a7-8f80-42b0-9655-a015b73f2857.png">

   You can also create registeries and manipluate them, by code : https://www.mlflow.org/docs/latest/model-registry.html
  
