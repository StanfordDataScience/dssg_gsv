# Notebooks

## Visualization: time-series slider
- `visualize_timeseries.ipynb`:  Used to plot prediction results onto map (particularly time series images). Allows for creation of map with predictions averaged by block group, and a slider for year. See documentation of plot_map() in the notebook for more details.

## Compare the regression results with classification
- `regr_to_class_transform_det_bos.ipynb`: Used to turn regression results back into classes to compare with previous models. Contains methods to plot regression results, overlay classification regions, and compare classification metrics.

## Other tested methods
- `autoencoder.ipynb`: Used to test an autoencoder-based anomaly detector for blight.
- `nonlinear_models.ipynb`: Used to test modifications to the ResNet-18 regression model, particularly one that replaced the linear "head" layer with a module that performs a quadratic function of the input feature vectors.

## Data Cleaning & Transformation
- `create_valcsv.ipynb`: Used to transform a folder full of image to val.csv, which can be used for testing purpose.
- `s3_azure.ipynb`: Used to move the data from azure ML dataset to azure workspace/notebook.
- `time-series.ipynb`: This notebook contains code for all the time series analysis. This is the pipeline for Detroit dataset and can be easily adapted for other cities. It contains various visualization and dataframe manipulation

## Other visualization notebooks
- `regression_results.ipynb`: Used to generate regression results from the experiment. Generates linear plot to demonstrate the relationship between the ground truth and the prediction.
- `visualize_trueskills_detroit&boston.ipynb`: Used to read and print the boston and detroit prediction images with the trueskill score, class number and index. Also including the visulization of the class distribution.
