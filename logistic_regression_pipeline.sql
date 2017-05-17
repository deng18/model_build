--------------------------------------------------------------------------------------------------------------------------
--                                  Model building : logistic regression pipeline                                       --
--                                       Xi Zhang <zhan2037@purdue.edu>                                                 --
--                                                 March 2016                                                           --
--------------------------------------------------------------------------------------------------------------------------
-- Need to change imputation function and store mean/variance/etc
-- Need to create scoring function

--------------------------------------------------------------------------------------------------------------------------
-- Apply a mask to a feature list to only extract selected features based on the mask
-- Ex: feature_vals = [1,3,4,5], mask = [0,1,1,0], selected features = [3,4]
create or replace function oir.select_features(feature_vals float8[], mask int[]) returns float8[] as
$$
	#Only return those indices whose values in the mask field are '1'
	retvals = [val for indx, val in enumerate(feature_vals) if mask[indx]] 
	return retvals
$$ language plpythonu;

-- Returns selected feature labels
create or replace function oir.select_features(feature_labels text[], mask int[]) returns text[] as
$$
	#Only return those indices whose values in the mask field are '1'
	retvals = [val for indx, val in enumerate(feature_labels) if mask[indx]] 
	return retvals
$$ language plpythonu;	

create or replace function oir.logit_roc_curve(schema text, model_name text) returns void as 
$$
	sql <- paste('select FPR as x, recall_P as y from ', schema, '.', model_name, '_logit_mdl_eval_metrics', sep='');
	str <- c(pg.spi.exec(sql));

	mymain <- 'ROC Curve';
	mysubtitle <- paste('Logistic regression for ', model_name, sep='')
	myxlab <- '1-Specificity';
	myylab <- 'Sensitivity';
	path <- paste('/tmp/', model_name, '_logit_roc.pdf', sep='')

	pdf(path);
	plot(str, type='b', main=mymain, xlab=myxlab, ylab=myylab, lwd=3);
	mtext(mysubtitle)
	dev.off();
$$ language plr;

create or replace function oir.logit_p_value_cdf(schema text, model_name text) returns void as 
$$
	sql <- paste('select p_value as x, cdf as y from ', schema, '.', model_name, '_logit_mdl_full_data_p_value_cdf', sep='');
	str <- c(pg.spi.exec(sql));

	mymain <- 'CDF Curve of P-Values';
	mysubtitle <- paste('Logistic regression for ', model_name, sep='')
	myxlab <- 'P-Value';
	myylab <- 'CDF Value';
	path <- paste('/tmp/', model_name, '_logit_p_value_cdf.pdf', sep='')

	pdf(path);
	plot(str, type='b', main=mymain, xlab=myxlab, ylab=myylab, lwd=3);
	mtext(mysubtitle)
	dev.off();
$$ language plr;

--------------------------------------------------------------------------------------------------------------------------
-- Parameter setting for logistic regression pipeline
-- schema: schema for both input and output tables.
-- model_name: unique name is recommened (Ex: grade_mod_bigclasses, at_risk_model_042416).
-- input_table: input table name, which should be located in the schema specified above.
-- key: unique identifier of input table.
-- class: dependent variable, binary response only, 1 is postive class, 0 is negative class.
-- exclusion_list: columns in the input table to be excluded from feature imputation and normalization, including non-features and reference levels for one-hot encoded features.
-- vif_steps: number of vif steps (max values is total number of independent variables you have).
-- perc_train: percent of training vs test split (Ex: 0.8 means 80% training set, 20% test set).
-- num_train_folds: number of folds to split training set for ensembling, it needs to be >= 2. Ex: 4 folds means model will be trained on 4 subsets of the training set and then the results will be combined.
-- pos_class_weight: 1 if positive and negative classes are weighted equally, greater than 1 means positive class has more weight.
--------------------------------------------------------------------------------------------------------------------------
create or replace function oir.logistic_regression_pipeline(schema text, model_name text, input_table text, key text, class text, exclusion_list text[], 
	vif_steps int, perc_train float8, num_train_folds int, pos_class_weight float8) returns text as
	
$BODY$

DECLARE 
deleted_num_rows int;
filtered_num_feats int;
tbl_not_empty int;
coef_check text;

BEGIN
--------------------------------------------------------------------------------------------------------------------------
-- 0) Data preprocessing: delete duplicate rows                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup as
	(
		select * from $$ || schema || $$.$$ || input_table || $$
	) distributed by ($$ || key || $$)$$;

EXECUTE $$select count(*) 
	from (select $$ || key || $$, row_number() over (partition by $$ || key || $$) as rnum from $$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup) t 
	where rnum >1$$
	into deleted_num_rows;

EXECUTE $$delete from $$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup 
	where $$ || key || $$ in 
	(
		select $$ || key || $$
        from (select $$ || key || $$, row_number() over (partition by $$ || key || $$) as rnum from $$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup) t
        where rnum > 1
    )$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1a) Prepare dataset for logistic regression                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed cascade$$;
EXECUTE $$select oir.impute_missing_values(
	'$$ || schema || $$', 
	'$$ || model_name || $$_logit_input_table_nodup',   --input table name
	array['$$ || array_to_string(exclusion_list, $a$', '$a$) || $$', '$$ || key || $$', '$$ || class || $$'], --columns to exclude
	'$$ || key || $$',  --id column
	'$$ || class || $$',   --label column
	TRUE, --whether to include constant term for intercept (if running regression)
	'$$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed'   --output table name
)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1b) Feature select using ViF. 
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$select oir.feature_selection_w_vif(
    '$$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed', --input table
    '$$ || key || $$', --id column
    'feat_name_vect', --features names (array) column
    'feat_vect_normalized', --feature values (array) column
    10, --ViF threshold (recommend >= 10),
    $$ || vif_steps || $$, --num_steps (max values is total number of independent variables you have). 
    '$$ || schema || $$.$$ || model_name || $$_logit_vif_results' --results table
)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1c) Remove features based on ViF results. 
--------------------------------------------------------------------------------------------------------------------------
-- Store the mask in a table.
EXECUTE $$drop table if exists $$ || model_name || $$_logit_vif_mask cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_vif_mask as
(
	select 
	    array_agg(mask order by indx) as mask_arr
	from
	(
	  	select 
	  		indx,
	        feat_name,
	        case when q2.feature_w_max_vif is null then 1 else 0 end as mask
	  	from
	  	(
	  		select 
	  			generate_series(1, array_upper(feat_name_vect, 1)) as indx,
	           	unnest(feat_name_vect) as feat_name
	    	from (select feat_name_vect from $$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed limit 1)foo
	  	)q1
	  	left join 
	  	$$ || schema || $$.$$ || model_name || $$_logit_vif_results q2
	  	on q1.feat_name = q2.feature_w_max_vif and q2.discarded = TRUE
	)t
) distributed randomly$$;

-- Apply mask in the features on imputed input table.
EXECUTE $$drop table if exists $$ || model_name || $$_logit_input_table_imputed_vif_filtered cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_input_table_imputed_vif_filtered as
(
	select 
	  	$$ || key || $$,
	    oir.select_features(feat_name_vect, mask_arr) as feat_name_vect,
	    oir.select_features(feat_vect, mask_arr) as feat_vect,
	    oir.select_features(feat_vect_normalized, mask_arr) as feat_vect_normalized,
	    $$ || class || $$
	from 
	  	$$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed t1,
	    $$ || model_name || $$_logit_vif_mask t2
) distributed by ($$ || key || $$)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 2) Prepare training & test sets. 
--------------------------------------------------------------------------------------------------------------------------
--Use a random seed to determine what samples go into training set & what samples into test set.
EXECUTE $$drop table if exists $$ || model_name || $$_logit_all_rows cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_all_rows as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || class || $$,
	    random() as splitter
	from 
		$$ || model_name || $$_logit_input_table_imputed_vif_filtered
	where 
		$$ || class || $$ is not null    -- only consider non-null values for the label
) distributed by ($$ || key || $$)$$;

--Training table
EXECUTE $$drop table if exists $$ || model_name || $$_logit_train cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_train as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || class || $$
	from 
		$$ || model_name || $$_logit_all_rows
	where 
		splitter <= $$ || perc_train || $$   --percent of training set
) distributed by ($$ || key || $$)$$;

--Testing table
EXECUTE $$drop table if exists $$ || model_name || $$_logit_test cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_test as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || class || $$
	from 
		$$ || model_name || $$_logit_all_rows
	where 
		splitter > $$ || perc_train || $$   
) distributed by ($$ || key || $$)$$;

-------------------------------------------------------------------------------------------------------------------------------
-- 3) Ensemble logistic regression models.
-- 3a) Split training set by i folds, create i samples, each sample contains (i-1)/i of the training data (throw out one fold). 
--------------------------------------------------------------------------------------------------------------------------------  
-- Create i samples
FOR i IN 1..num_train_folds LOOP
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_train_sample$$ || i || $$ cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_train_sample$$ || i || $$ as 
	(
		select 
			$$ || key || $$, 
	       	feat_name_vect,
	       	feat_vect,
	      	feat_vect_normalized, 
		    $$ || class || $$ 
		from 
		(
			select *, floor(random()*$$ || num_train_folds || $$) as splitter 
			from $$ || model_name || $$_logit_train
		)t
		where
			splitter != $$ || i || $$-1
	) distributed by ($$ || key || $$)$$;

END LOOP;

FOR i IN 1..num_train_folds LOOP
	-------------------------------------------------------------------------------------------------------------------------------
	-- 3b) Build Logistic Regression model on each sample. 
	--------------------------------------------------------------------------------------------------------------------------------  
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$ cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$_summary cascade$$;
	EXECUTE $$select madlib.logregr_train( 
		'$$ || model_name || $$_logit_train_sample$$ || i ||$$',
		'$$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$',
		'$$ || class || $$',
		'feat_vect_normalized',
		NULL,
		20,
		'irls'
	)$$;

	EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i 
		into coef_check;

	WHILE coef_check is null LOOP
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$ cascade$$;
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$_summary cascade$$;
		EXECUTE $$select madlib.logregr_train( 
			'$$ || model_name || $$_logit_train_sample$$ || i ||$$',
			'$$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$',
			'$$ || class || $$',
			'feat_vect_normalized',
			NULL,
			20,
			'irls'
		)$$;

		EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i 
			into coef_check;

	END LOOP;
	--------------------------------------------------------------------------------------------------------------------------
	-- 3c) Filter out junk features based on p-value and retrain the model with the selected features.
	-- We'll use filtering based feature selection on account of lower complexity & time. If we had more time, we'd implement
	-- wrapper based feature selection algorithms.
	-- Refer: http://www.mathworks.com/help/stats/examples/selecting-features-for-classifying-high-dimensional-data.html
	--------------------------------------------------------------------------------------------------------------------------  
	-- Store the mask in a table.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_mdl_mask_sample$$ || i || $$ cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_mdl_mask_sample$$ || i || $$ as
	(
		select 
			array_agg(coef order by indx) as coef_arr,
		  	array_agg(p_value order by indx) as p_value_arr,
		    array_agg(mask order by indx) as mask_arr
		from
		(
		  	select 
		  		indx,
		        coef,
		        p_value,
		        --select only those features whose p-values are < 0.05
		        case when p_value < 0.05 then 1 else 0 end as mask
		  	from
		  	(
		  		select 
		  			generate_series(1, array_upper(coef, 1)) as indx,
		           	unnest(coef) as coef,
		           	unnest(p_values) as p_value
		    	from $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$
		  	)q1
		)q2
	) distributed randomly$$;

	--------------------------------------------------------------------------------------------------------------------------
	-- 3d) Re-train the model while filtering out junk features using the mask
	--------------------------------------------------------------------------------------------------------------------------  
	-- Apply mask in the features on training set.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_train_filtered_sample$$ || i || $$ cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_train_filtered_sample$$ || i || $$ as
	(
		select 
		  	$$ || key || $$,
		    oir.select_features(feat_name_vect, mask_arr) as feat_name_vect,
		    oir.select_features(feat_vect_normalized, mask_arr) as feat_vect_normalized,
		    $$ || class || $$
		from 
		  	$$ || model_name || $$_logit_train_sample$$ || i || $$ t1,
		    $$ || model_name || $$_logit_mdl_mask_sample$$ || i || $$ t2
	) distributed by ($$ || key || $$)$$;

	-- Apply mask on features in test set.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_test_filtered_sample$$ || i || $$ cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_test_filtered_sample$$ || i || $$ as
	(
		select 
		  	$$ || key || $$,
		    oir.select_features(feat_name_vect, mask_arr) as feat_name_vect,
		    oir.select_features(feat_vect_normalized, mask_arr) as feat_vect_normalized,
		    $$ || class || $$
		from 
		  	$$ || model_name || $$_logit_test t1,
		    $$ || model_name || $$_logit_mdl_mask_sample$$ || i || $$ t2
	) distributed by ($$ || key || $$)$$; 

	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$ cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_sample$$ || i || $$_summary cascade$$;

END LOOP;

EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_feature_rank$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_feature_rank(
	selected_feature text,
	p_value float8,
	coef float8,
	sample int,
	relative_rank float8) distributed randomly$$;

EXECUTE $$drop table if exists $$ || model_name || $$_logit_test_pred_prob$$;
EXECUTE $$create temp table $$ || model_name || $$_logit_test_pred_prob(
	$$ || key || $$ text,
	$$ || class || $$ text,
	pred_prob float8,
	sample int) distributed by ($$ || key || $$)$$;

<<retrain>>
FOR i IN 1..num_train_folds LOOP

	EXECUTE $$select array_upper(feat_vect_normalized, 1) from $$ || model_name || $$_logit_train_filtered_sample$$ || i
		INTO filtered_num_feats;
 	
 	CONTINUE retrain WHEN filtered_num_feats = 0;

	-- Re-train the model while filtering out junk features using the mask
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$ cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$_summary cascade$$;
	EXECUTE $$select madlib.logregr_train( 
		'$$ || model_name || $$_logit_train_filtered_sample$$ || i || $$',
		'$$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$',
		'$$ || class || $$',
		'feat_vect_normalized',
		NULL,
		20,
		'irls'
	)$$;  

	EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i 
		into coef_check;

	WHILE coef_check is null LOOP
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$ cascade$$;
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$_summary cascade$$;
		EXECUTE $$select madlib.logregr_train( 
			'$$ || model_name || $$_logit_train_filtered_sample$$ || i || $$',
			'$$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$',
			'$$ || class || $$',
			'feat_vect_normalized',
			NULL,
			20,
			'irls'
		)$$;

		EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i 
			into coef_check;

	END LOOP;
	--------------------------------------------------------------------------------------------------------------------------
	-- 3e) Calculate relative ranks of selected features.
	-- http://www.mwsug.org/proceedings/2009/stats/MWSUG-2009-D10.pdf
	--------------------------------------------------------------------------------------------------------------------------  
	EXECUTE $$
		insert into $$ || schema || $$.$$ || model_name || $$_logit_feature_rank
		select *, cume_dist() over(order by abs(coef) asc) as relative_rank
		from
		(
			select 
				unnest(feat_name_vect) as selected_feature,
			    unnest(p_values) as p_value,
			    unnest(coef) as coef,
			    '$$ || i || $$' as sample
			from 
				(
					select *
					from $$ || model_name || $$_logit_test_filtered_sample$$ || i || $$
					limit 1
				)t1,
				$$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$ t2
		)tbl
		where selected_feature != 'constant_term' and p_value < 0.05$$;

	--------------------------------------------------------------------------------------------------------------------------
	-- 3f) Test predictions on the test set with the trained logistic regression model.
	--------------------------------------------------------------------------------------------------------------------------  
	EXECUTE $$
		insert into $$ || model_name || $$_logit_test_pred_prob
		select 
	    	$$ || key || $$,
	      	$$ || class || $$,
	      	madlib.logregr_predict_prob(coef, feat_vect_normalized) as pred_prob,
	      	$$ || i || $$::int as sample
	  	from 
	    	$$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$ t1,
	      	$$ || model_name || $$_logit_test_filtered_sample$$ || i || $$ t2$$;

	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$ cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_filtered_sample$$ || i || $$_summary cascade$$;

END LOOP retrain;

EXECUTE $$select count(*) from (select 1 from $$ || schema || $$.$$ || model_name || $$_logit_feature_rank limit 1) t$$ into tbl_not_empty;

IF tbl_not_empty = 0 THEN 
	EXECUTE $$DROP TABLE IF EXISTS 
		$$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup,
		$$ || schema || $$.$$ || model_name || $$_logit_feature_rank,
		$$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed$$;

	RAISE NOTICE 'None of the features have p-value lower than 0.05 for all iterations';
	RETURN 'ABORTED';
	EXIT;

ELSE
	--------------------------------------------------------------------------------------------------------------------------
	-- 4) Calculate feature importance scores (average relative ranks) of selected features.
	-------------------------------------------------------------------------------------------------------------------------- 
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_feature_importance cascade$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_feature_importance as
	(
		select
			selected_feature,
			avg(relative_rank) as feature_importance
		from 
			$$ || schema || $$.$$ || model_name || $$_logit_feature_rank
		group by 1
		order by 2 desc
	) distributed randomly$$;

	--------------------------------------------------------------------------------------------------------------------------
	-- 5) Calculate model performance metrics.
	--------------------------------------------------------------------------------------------------------------------------  
	-- Calculate weighted average probabilities for each sample.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_test_avg_pred_prob cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_test_avg_pred_prob as
	(
		select 
	    	$$ || key || $$,
	      	$$ || class || $$,
	      	avg(pred_prob) * $$ || pos_class_weight || $$ as avg_pred_prob
	    from
	    	$$ || model_name || $$_logit_test_pred_prob
	    group by 1,2
	) distributed by ($$ || key || $$)$$;

	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_test_pred_class cascade$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_test_pred_class as
	(
		select *, case when avg_pred_prob > threshold then 1 else 0 end as pred_class
		from
		(
			select t.*, threshold*0.01 as threshold
			from generate_series(0, 100) threshold
			cross join $$ || model_name || $$_logit_test_avg_pred_prob t
		)tbl
	) distributed by ($$ || key || $$)$$;

	-- Precision, Recall & F1 Score.
	-- Plot FPR (x-axis) and recall_P (y-axis) for ROC curve.
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_eval_metrics$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_mdl_eval_metrics as
	(
		select 
			threshold,
		  	precision_P,
		 	recall_P,
			2.0*precision_P*recall_P/((precision_P+recall_P)+0.001*(precision_P+recall_P=0)::int) as f1_score_P,
		  	precision_N,
		  	recall_N,
		 	2.0*precision_N*recall_N/((precision_N+recall_N)+0.001*(precision_N+recall_N=0)::int) as f1_score_N,
		 	FPR
		from
		(
		    select 
		    	threshold,
			    sum(TP)*1.0/(sum(P)+0.001*(sum(P)=0)::int) as recall_P,  --TPR 
			    sum(TN)*1.0/(sum(N)+0.001*(sum(N)=0)::int) as recall_N,   
			    sum(TP)*1.0/(sum(pred_P)+0.001*(sum(pred_P)=0)::int) as precision_P,   
			    sum(TN)*1.0/(sum(pred_N)+0.001*(sum(pred_N)=0)::int) as precision_N,   
			    sum(FP)*1.0/(sum(N)+0.001*(sum(N)=0)::int) as FPR   
		    from
		    (
			    select 
			    	threshold,
			      	case when ($$ || class || $$ = 1) and (pred_class = 1) then 1 else 0 end as TP,
			      	case when ($$ || class || $$ = 0) and (pred_class = 0) then 1 else 0 end as TN,
			      	case when ($$ || class || $$ = 0) and (pred_class = 1) then 1 else 0 end as FP,
			      	case when ($$ || class || $$ = 1) then 1 else 0 end as P,
			     	case when ($$ || class || $$ = 0) then 1 else 0 end as N,
			      	case when (pred_class = 1) then 1 else 0 end as pred_P,
			      	case when (pred_class = 0) then 1 else 0 end as pred_N
			    from $$ || schema || $$.$$ || model_name || $$_logit_test_pred_class
		    )q1 group by 1
		)q2 order by 1 asc
	) distributed randomly$$;

	-- Generate ROC curve, which will be saved at /tmp/<model_name>_logit_roc.pdf
	--EXECUTE $$select oir.logit_roc_curve('$$ || schema || $$', '$$ || model_name || $$')$$; 

	--------------------------------------------------------------------------------------------------------------------------
	-- 6) Re-train the model on full dataset to get the signs of coeffcients
	--------------------------------------------------------------------------------------------------------------------------  
	-- Prepare dataset for final logistic regression on full data                   
	-- Store the mask in a table.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_final_feat_list_mask cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_final_feat_list_mask as
	(
		select 
		    array_agg(mask order by indx) as mask_arr
		from
		(
		  	select 
		  		indx,
		        feat_name,
		        case when q2.selected_feature is not null then 1 else 0 end as mask
		  	from
		  	(
		  		select 
		  			generate_series(1, array_upper(feat_name_vect, 1)) as indx,
		           	unnest(feat_name_vect) as feat_name
		    	from (select feat_name_vect from $$ || model_name || $$_logit_input_table_imputed_vif_filtered limit 1)foo
		  	)q1
		  	left join 
		  	$$ || schema || $$.$$ || model_name || $$_logit_feature_importance q2
		  	on q1.feat_name = q2.selected_feature
		)t
	) distributed randomly$$;

	-- Apply mask in the features on imputed and vif filtered input table.
	EXECUTE $$drop table if exists $$ || model_name || $$_logit_full_data_filtered cascade$$;
	EXECUTE $$create temp table $$ || model_name || $$_logit_full_data_filtered as
	(
		select 
		  	$$ || key || $$,
		    oir.select_features(feat_name_vect, mask_arr) as feat_name_vect,
		    oir.select_features(feat_vect, mask_arr) as feat_vect,
		    oir.select_features(feat_vect_normalized, mask_arr) as feat_vect_normalized,
		    $$ || class || $$
		from 
		  	$$ || model_name || $$_logit_input_table_imputed_vif_filtered t1,
		    $$ || model_name || $$_logit_final_feat_list_mask t2
	) distributed by ($$ || key || $$)$$;

	-- Re-train the model on full dataset.
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_summary cascade$$;
	EXECUTE $$select madlib.logregr_train( 
		'$$ || model_name || $$_logit_full_data_filtered',
		'$$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data',
		'$$ || class || $$',
		'feat_vect_normalized',
		NULL,
		20,
		'irls'
	)$$;

	EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data$$ 
		into coef_check;

	WHILE coef_check is null LOOP
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data cascade$$;
		EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_summary cascade$$;
		EXECUTE $$select madlib.logregr_train( 
			'$$ || model_name || $$_logit_full_data_filtered',
			'$$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data',
			'$$ || class || $$',
			'feat_vect_normalized',
			NULL,
			20,
			'irls'
		)$$;

		EXECUTE $$select coef from $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data$$ 
			into coef_check;

	END LOOP;

	-- Top-20 +ve coefficients
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_top20_pos_coef cascade$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_top20_pos_coef as 
	(
		select 
			selected_feature,
		    round(coefficient::numeric, 2) as coefficient,
		    round(p_value::numeric, 2) as p_value
		from
		(
			select 
				unnest(feat_name_vect) as selected_feature,
			    unnest(coef) as coefficient,
			    unnest(p_values) as p_value
			from
				(
					select * 
					from $$ || model_name || $$_logit_full_data_filtered
					limit 1
				)t1,
				$$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data t2
		)tbl
		where selected_feature != 'constant_term' and p_value < 0.05 and coefficient > 0
		order by coefficient desc
		limit 20	
	) distributed randomly$$;

	-- Top-20 -ve coefficients
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_top20_neg_coef cascade$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_top20_neg_coef as 
	(
		select 
			selected_feature,
		    round(coefficient::numeric, 2) as coefficient,
		    round(p_value::numeric, 2) as p_value
		from
		(
			select 
				unnest(feat_name_vect) as selected_feature,
			    unnest(coef) as coefficient,
			    unnest(p_values) as p_value
			from 
				(
					select * 
					from $$ || model_name || $$_logit_full_data_filtered
					limit 1
				)t1,
				$$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data t2
		)tbl
		where selected_feature != 'constant_term' and p_value < 0.05 and coefficient < 0
		order by coefficient asc
		limit 20	
	) distributed randomly$$;

	-- Show CDF of p-values with feature counts
	-- Refer: http://www.mathworks.com/help/stats/examples/selecting-features-for-classifying-high-dimensional-data.html
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_p_value_cdf cascade$$;
	EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data_p_value_cdf as
	(
		select 
			p_value,
		    (count(*) over(order by p_value))*1.0
			/
	       	(
				select array_upper(p_values, 1) as num_elements
				from $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data
	       	) as cdf
		from
		(
			select p_value, count(*) as freq
			from
			(
				select unnest(p_values) as p_value
				from $$ || schema || $$.$$ || model_name || $$_logit_mdl_full_data
			)q
			group by 1
		)tbl
		order by p_value asc
	) distributed randomly$$;

	-- Generate CDF curve of p-values, which will be saved at /tmp/<model_name>_logit_p_value_cdf.pdf
	--EXECUTE $$select oir.logit_p_value_cdf('$$ || schema || $$', '$$ || model_name || $$')$$; 

	EXECUTE $$DROP TABLE IF EXISTS 
	$$ || schema || $$.$$ || model_name || $$_logit_input_table_nodup,
	$$ || schema || $$.$$ || model_name || $$_logit_feature_rank,
	$$ || schema || $$.$$ || model_name || $$_logit_input_table_imputed$$;

	--RAISE NOTICE USING MESSAGE = 'Deleted number of rows with duplicated keys: ' || deleted_num_rows;
	RETURN 'DONE';
END IF;

END;
$BODY$
LANGUAGE PLPGSQL;

-----------------------------------------------------------------------------------------------------
-- Sample invocations
-----------------------------------------------------------------------------------------------------
/*
select oir.logistic_regression_pipeline(
	'xz',   --output schema
	'old_mod_20_apr_2016_sample',  --model name
	'grade_mod_input_table_4terms',   --input table name
	'pk1',   --unique identifier
	'grade_2class',   --class column
	ARRAY[
	'person_uid',
	'academic_period',
	'course_id',
	'grade_value',
	'pk1',
	'grade_2class',
	'grade_2class_desc',
	'grade_3class',
	'grade_3class_desc',
	--Remove features which have high correlation with other features
	'board_plan_bp_008w_up_a',
	'brd_pln_dining_usage_prob_hr_of_day_12',
	'class_boap_04',
	'door_access_prob_hr_of_day_12',
	'enroll_college_a',
	'gender_f',
	'housing_building_code_tark',
	'profile_admissions_population_none',
	'profile_college_a',
	'rectrac_login_dow_thu',
	'reporting_ethnicity_unknown',
	'strd_val_tran_dining_usage_prob_hr_of_day_12',
	'strd_val_tran_laundry_usage_prob_hr_of_day_12',
	'drop_days_after_class_begin',
	'drop_days_before_ticket_end',
	'course_reg_duration',
	'schedule_ind',
	'class_hr_of_day_12',
	'offering_college_a',
	--'num_days_start_reg_after_ticket_begin',
	'add_days_after_class_begin',
	'drop_days_after_class_begin',
	'drop_days_before_ticket_end',
	'course_reg_duration',
	--'min_num_days_start_reg_after_ticket_begin',
	--'avg_add_days_after_ticket_begin',
	'avg_add_days_after_class_begin',
	'avg_drop_days_after_class_begin',
	'avg_drop_days_before_ticket_end',
	'avg_course_reg_duration'
	],
	5,
	0.8, 
	4, 
	1
	);
	*/