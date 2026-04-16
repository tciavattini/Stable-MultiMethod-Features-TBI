1. 01_feature_engineering_pipeline.ipynb --> code to run feature feature engineering
2. 02_outcome classification.py --> to classify the patients based on quantile
3. 03_prepare_data.py --> to prepare data for models
4. sanity check --> run before modelling
5. run cross_validation.py in this way (usando il codice stability_top30.py)
    nohup python stability_top30.py > logs/stability_run_$(date +%Y%m%d_%H%M).log 2>&1 &
    echo $! > logs/stability.pid

    to run cross_validation you need the files data_loading.py and stability_top30
    
baseline_characteristics.py --> code to reproduce the first table
