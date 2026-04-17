1. 01_feature_engineering_pipeline.ipynb --> code to run feature feature engineering
2. 02_outcome classification.py --> to classify the patients based on quantile
3. 03_prepare_data.py --> to prepare data for models
4. sanity check --> run before modelling
5. run 04_cstability_top30.py in this way (usando il codice stability_top30.py)
    nohup python stability_top30.py > logs/stability_run_$(date +%Y%m%d_%H%M).log 2>&1 &
    echo $! > logs/stability.pid

    to run cross_validation you need the files data_loading.py and stability_top30

6. run 05_multifeature.py --> multifeature analysis. in the code it's present also the code to generate figure 4b, and table S11 with the name statistical_comparison.csv, file publication_table.csv used 
for table 2 of main paper, feature_count_analysis is table S10 from supplementary

baseline_characteristics.py --> code to reproduce the first table

7. 06_alt_feature_selection.py --> serve per fare la feature selection con i tre metodi alternativi e da in risultato le feature di quei tre metodi anche combinati

8. 07_alt_fs_classification.py --> part of the results of file classification_performance.csv is in the table 2 of main paper

9. 08_phenotype_analysis.py

10. 09_sensitivity_analyses.py
