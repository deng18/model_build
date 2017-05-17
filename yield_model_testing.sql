select
	ip.xgboost_grid_search(
		'oir',--training table_schema
		'yield_model_201510_asof_1_26',--training table_name
		'person_uid', -- id column
		'matriculated', -- class label column
		-- Columns to exclude from features (independent variables)
		ARRAY[
			'person_uid', 
			'matriculated',
			'college'
		],
		--XGBoost grid search parameters
		$$
		{
			'learning_rate': [0.005], #Regularization on weights (eta). For smaller values, increase n_estimators
			'max_depth': [10],#Larger values could lead to overfitting
			'subsample': [0.8],#introduce randomness in samples picked to prevent overfitting
			'colsample_bytree': [0.55],#introduce randomness in features picked to prevent overfitting
			'min_child_weight': [1],#larger values will prevent over-fitting
			'n_estimators':[1000] #More estimators, lesser variance (better fit on test set)
		}
		$$,
		--Grid search parameters temp table (will be dropped when session ends)
		'ip_grid_search_params_temp_tbl',
		--Grid search results table.
		'ip.grid_search_mdl_results',
		--class weights (set it to empty string '' if you want it to be automatic)
		''
	);

--show the results of the various models
select metrics, params from ip.grid_search_mdl_results;

select
	ip.xgboost_mdl_score(
		'oir.yield_model_201610_asof_1_26', -- scoring table
		'person_uid', -- id column
		'matriculated', -- class label column, NULL if unavailable
		'ip.grid_search_mdl_results', -- model table
		'True', -- model filter, set to 'True' if no filter
		'ip.xgboost_best_mdl_results'
	);

select metrics, params from ip.xgboost_best_mdl_results;
