# MLflow for begginers

## In this tutorial, you'll learn how to manage a Machine Learning project lifecycle with MLflow

### Install MLflow on linux environement

    pip install mlflow[extras]

### Explore the "titanic_classification.ipynb" notebook
   This is a fairly simple classification example. 
   It illustrates how to apply different preprocessing and feature extraction pipelines.
   
   For more details check the link below :
        
        https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#column-transformer-with-mixed-types
### Launch MLflow on localhost with Tracking Server:

    mlflow ui --backend-store-uri sqlite:///mlruns.db
