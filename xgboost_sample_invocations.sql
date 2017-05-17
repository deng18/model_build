-----------------------------------------------------------------------------------------------------
-- XGBoost grid search pipeline sample invocations
-----------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
--                                  GRADE MODEL
-------------------------------------------------------------------------------------
select
    oir.xgboost_grid_search(
        'oir',--training table_schema
        'grade_mod_input_table_26_apr_2016_bigclasses',--training table_name
        'pk1', -- id column
        'grade_2class', -- class label column
        -- Columns to exclude from features (independent variables)
        ARRAY[
            'puid',
            'person_uid', 
            'academic_period',
            'pk1',
            'course_identification',
            'course_gpa',
            'grade_3class',
            'grade_3class_desc',
            'grade_2class',
            'grade_2class_desc'
        ],
        --XGBoost grid search parameters
        $$
        {
            'learning_rate': [0.01], #Regularization on weights (eta). For smaller values, increase n_estimators
            'max_depth': [16,18],#Larger values could lead to overfitting
            'subsample': [0.8],#introduce randomness in samples picked to prevent overfitting
            'colsample_bytree': [0.65],#introduce randomness in features picked to prevent overfitting
            'min_child_weight': [1],#larger values will prevent over-fitting
            'n_estimators': [600] #More estimators, lesser variance (better fit on test set)
        }
        $$,
        --Grid search parameters temp table (will be dropped when session ends)
        'xgb_params_temp_tbl',
        --Grid search results table.
        'oir.grade_grid_search_mdl_results_90_perc_train',
        --class weights (set it to empty string '' if you want it to be automatic)
        '',
        0.9
    );

--metrics for negative class (gpa under 2.0)
select  
    params, 
    params_indx, 
    substring(metrics from 81 for 8) as neg_fscore,
    substring(metrics from 61 for 8) as neg_precision,
    substring(metrics from 71 for 8) as neg_recall
from oir.grade_grid_search_mdl_results_90_perc_train
order by neg_fscore desc;

--data comparision sorted by feature importance
select 
    t1.*,
    t2.imp::int as imp
from
    oir.grade_mod_201610_vs_2015_comp t1
left outer join
    (
    select unnest(f_importances) as imp, unnest(fnames) as names
    from
    oir.grade_grid_search_mdl_results
    where params_indx = 1
    ) t2
on (t1.col_name=t2.names)
where imp is not null
order by imp desc

