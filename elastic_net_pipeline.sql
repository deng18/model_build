--------------------------------------------------------------------------------------------------------------------------
--                                  Model building : elastic net pipeline                                               --
--                                       Xi Zhang <zhan2037@purdue.edu>                                                 --
--                                                 June 2016                                                            --
--------------------------------------------------------------------------------------------------------------------------
-- Need to add class label to scoring function
-- Need to change imputation function and store mean/variance/etc

--------------------------------------------------------------------------------------------------------------------------
-- Parameter setting for elastic net pipeline
-- schema: schema for both input and output tables.
-- model_name: unique name is recommened (Ex: grade_mod_bigclasses, at_risk_model_042416).
-- input_table: input table name, which should be located in the schema specified above.
-- key: unique identifier of input table.
-- dep_var: dependent variable, continuous variable.
-- exclusion_list: columns in the input table to be excluded from feature imputation and normalization, including non-features and reference levels for one-hot encoded features.
-- vif_steps: number of vif steps (max values is total number of independent variables you have).
-- perc_train: percent of training vs test split (Ex: 0.8 means 80% training set, 20% test set).
--------------------------------------------------------------------------------------------------------------------------
create or replace function oir.elastic_net_pipeline(schema text, model_name text, input_table text, key text, dep_var text, exclusion_list text[], 
	vif_steps int, perc_train float8) returns text as
	
$BODY$

DECLARE 
coef_check text;

BEGIN
--------------------------------------------------------------------------------------------------------------------------
-- 0) Data preprocessing: delete duplicate rows                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_nodup cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_nodup as
	(
		select * from $$ || schema || $$.$$ || input_table || $$
	) distributed by ($$ || key || $$)$$;

EXECUTE $$delete from $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_nodup 
	where $$ || key || $$ in 
	(
		select $$ || key || $$
        from (select $$ || key || $$, row_number() over (partition by $$ || key || $$) as rnum from $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_nodup) t
        where rnum > 1
    )$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1a) Prepare dataset for elastic net                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed cascade$$;
EXECUTE $$select oir.impute_missing_values(
	'$$ || schema || $$', 
	'$$ || model_name || $$_elastnet_input_table_nodup',   --input table name
	array['$$ || array_to_string(exclusion_list, $a$', '$a$) || $$', '$$ || key || $$', '$$ || dep_var || $$'], --columns to exclude
	'$$ || key || $$',  --id column
	'$$ || dep_var || $$',   --label column
	TRUE, --whether to include constant term for intercept (if running regression)
	'$$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed'   --output table name
)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1b) Feature select using ViF. 
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$select oir.feature_selection_w_vif(
    '$$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed', --input table
    '$$ || key || $$', --id column
    'feat_name_vect', --features names (array) column
    'feat_vect_normalized', --feature values (array) column
    10, --ViF threshold (recommend >= 10),
    $$ || vif_steps || $$, --num_steps (max values is total number of independent variables you have). 
    '$$ || schema || $$.$$ || model_name || $$_elastnet_vif_results' --results table
)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 1c) Remove features based on ViF results. 
--------------------------------------------------------------------------------------------------------------------------
-- Store the mask in a table.
EXECUTE $$drop table if exists $$ || model_name || $$_elastnet_vif_mask cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_elastnet_vif_mask as
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
	    	from (select feat_name_vect from $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed limit 1)foo
	  	)q1
	  	left join 
	  	$$ || schema || $$.$$ || model_name || $$_elastnet_vif_results q2
	  	on q1.feat_name = q2.feature_w_max_vif and q2.discarded = TRUE
	)t
) distributed randomly$$;

-- Apply mask in the features on imputed input table.
EXECUTE $$drop table if exists $$ || model_name || $$_elastnet_input_table_imputed_vif_filtered cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_elastnet_input_table_imputed_vif_filtered as
(
	select 
	  	$$ || key || $$,
	    oir.select_features(feat_name_vect, mask_arr) as feat_name_vect,
	    oir.select_features(feat_vect, mask_arr) as feat_vect,
	    oir.select_features(feat_vect_normalized, mask_arr) as feat_vect_normalized,
	    $$ || dep_var || $$
	from 
	  	$$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed t1,
	    $$ || model_name || $$_elastnet_vif_mask t2
) distributed by ($$ || key || $$)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 2) Prepare training & test sets. 
--------------------------------------------------------------------------------------------------------------------------
--Use a random seed to determine what samples go into training set & what samples into test set.
EXECUTE $$drop table if exists $$ || model_name || $$_elastnet_all_rows cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_elastnet_all_rows as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || dep_var || $$,
	    random() as splitter
	from 
		$$ || model_name || $$_elastnet_input_table_imputed_vif_filtered
	where 
		$$ || dep_var || $$ is not null    -- only consider non-null values for the label
) distributed by ($$ || key || $$)$$;

--Training table
EXECUTE $$drop table if exists $$ || model_name || $$_elastnet_train cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_elastnet_train as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || dep_var || $$
	from 
		$$ || model_name || $$_elastnet_all_rows
	where 
		splitter <= $$ || perc_train || $$   --percent of training set
) distributed by ($$ || key || $$)$$;

--Testing table
EXECUTE $$drop table if exists $$ || model_name || $$_elastnet_test cascade$$;
EXECUTE $$create temp table $$ || model_name || $$_elastnet_test as
(
	select 
		$$ || key || $$, 
       	feat_name_vect,
       	feat_vect,
      	feat_vect_normalized, 
	    $$ || dep_var || $$
	from 
		$$ || model_name || $$_elastnet_all_rows
	where 
		splitter > $$ || perc_train || $$   
) distributed by ($$ || key || $$)$$;

-------------------------------------------------------------------------------------------------------------------------------
-- 3) Build elastic net model 
--------------------------------------------------------------------------------------------------------------------------------  
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl cascade$$;
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_summary cascade$$;
EXECUTE $$select madlib.elastic_net_train( 
	'$$ || model_name || $$_elastnet_train',
	'$$ || schema || $$.$$ || model_name || $$_elastnet_mdl',
	'$$ || dep_var || $$',
	'feat_vect_normalized',
    'gaussian',
    --Alpha (Lower alpha gives more weightage to Ridge Regression than Lasso)
    0.05,
    --Lambda (Larger lambda leads to sparse features)
    0.05,
    FALSE,
    NULL,
    --Optimizer
    'fista',
    '',
    NULL,
    10000,
    1e-6
)$$;

EXECUTE $$select coef_all from $$ || schema || $$.$$ || model_name || $$_elastnet_mdl$$ 
	into coef_check;

WHILE coef_check is null LOOP
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl cascade$$;
	EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_summary cascade$$;
	EXECUTE $$select madlib.elastic_net_train( 
		'$$ || model_name || $$_elastnet_train',
		'$$ || schema || $$.$$ || model_name || $$_elastnet_mdl',
		'$$ || dep_var || $$',
		'feat_vect_normalized',
        'gaussian',
        --Alpha (Lower alpha gives more weightage to Ridge Regression than Lasso)
        0.05,
        --Lambda (Larger lambda leads to sparse features)
        0.05,
        FALSE,
        NULL,
        --Optimizer
        'fista',
        '',
        NULL,
        10000,
        1e-6
	)$$;

	EXECUTE $$select coef_all from $$ || schema || $$.$$ || model_name || $$_elastnet_mdl$$
		into coef_check;

END LOOP;

--------------------------------------------------------------------------------------------------------------------------
-- 4) Calculate model performance metrics.
--------------------------------------------------------------------------------------------------------------------------  
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_test_pred cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_elastnet_test_pred as
(
    with temp as
    (
    	select 
    		$$ || key || $$,
    	  	actual_$$ || dep_var || $$,
    	  	pred_$$ || dep_var || $$,
    	  	actual_$$ || dep_var || $$ - pred_$$ || dep_var || $$ as residual
      	from
      	(
    		select 
    			$$ || key || $$,
    		  	$$ || dep_var || $$ as actual_$$ || dep_var || $$,
    		  	madlib.elastic_net_predict('gaussian', coef_all, intercept, feat_vect_normalized) as pred_$$ || dep_var || $$
    		from 
    			$$ || schema || $$.$$ || model_name || $$_elastnet_mdl t1,
    		  	$$ || model_name || $$_elastnet_test t2
    	)tbl
    )
    select *,
    	(select (1.0 - pow(stddev(residual),2)/pow(stddev(actual_$$ || dep_var || $$),2)) from temp) as r_square
    from temp
) distributed by ($$ || key || $$)$$;

-- Top-20 +ve coefficients
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_top20_pos_coef cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_top20_pos_coef as 
(
	select 
		selected_feature,
	    round(coefficient::numeric, 2) as coefficient
	from
	(
		select 
			unnest(feat_name_vect) as selected_feature,
		    unnest(coef_all) as coefficient
        from
			(
				select * 
				from $$ || model_name || $$_elastnet_test
				limit 1
			)t1,
			$$ || schema || $$.$$ || model_name || $$_elastnet_mdl t2
	)tbl
	where coefficient > 0
	order by coefficient desc
	limit 20	
) distributed randomly$$;

-- Top-20 -ve coefficients
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_top20_neg_coef cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || model_name || $$_elastnet_mdl_top20_neg_coef as 
(
	select 
		selected_feature,
	    round(coefficient::numeric, 2) as coefficient
	from
	(
		select 
			unnest(feat_name_vect) as selected_feature,
		    unnest(coef_all) as coefficient
		from 
			(
				select * 
				from $$ || model_name || $$_elastnet_test
				limit 1
			)t1,
			$$ || schema || $$.$$ || model_name || $$_elastnet_mdl t2
	)tbl
	where coefficient < 0
	order by coefficient asc
	limit 20	
) distributed randomly$$;

EXECUTE $$DROP TABLE IF EXISTS 
$$ || schema || $$.$$ || model_name || $$_elastnet_input_table_nodup,
$$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed$$;

RETURN 'DONE';

END;
$BODY$
LANGUAGE PLPGSQL;

--------------------------------------------------------------------------------------------------------------------------
-- 5) Scoring function. 
--------------------------------------------------------------------------------------------------------------------------
create or replace function oir.elastic_net_score(schema text, scoring_table text, model_table text, output_table text, key text, exclusion_list text[]) returns void as    
$BODY$
BEGIN

--------------------------------------------------------------------------------------------------------------------------
-- 4a) Data preprocessing: delete duplicate rows                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || scoring_table || $$_nodup cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || scoring_table || $$_nodup as
    (
        select * from $$ || schema || $$.$$ || scoring_table || $$
    ) distributed by ($$ || key || $$)$$;

EXECUTE $$delete from $$ || schema || $$.$$ || scoring_table || $$_nodup 
    where $$ || key || $$ in 
    (
        select $$ || key || $$
        from (select $$ || key || $$, row_number() over (partition by $$ || key || $$) as rnum from $$ || schema || $$.$$ || scoring_table || $$_nodup) t
        where rnum > 1
    )$$;

--------------------------------------------------------------------------------------------------------------------------
-- 4b) Prepare dataset for elastic net                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || model_name || $$_elastnet_input_table_imputed cascade$$;
EXECUTE $$select oir.impute_missing_values(
    '$$ || schema || $$', 
    '$$ || scoring_table || $$_nodup',   --input table name
    array['$$ || array_to_string(exclusion_list, $a$', '$a$) || $$', '$$ || key || $$', '$$ || dep_var || $$'], --columns to exclude
    '$$ || key || $$',  --id column
    '',   --label column
    TRUE, --whether to include constant term for intercept (if running regression)
    '$$ || schema || $$.$$ || scoring_table || $$_imputed'   --output table name
)$$;

--------------------------------------------------------------------------------------------------------------------------
-- 4c) Model scoring                     
--------------------------------------------------------------------------------------------------------------------------
EXECUTE $$drop table if exists $$ || schema || $$.$$ || output_table cascade$$;
EXECUTE $$create table $$ || schema || $$.$$ || output_table || $$ as
(
    select 
        $$ || key || $$,
        madlib.elastic_net_predict('gaussian', coef_all, intercept, feat_vect_normalized) as prediction
    from 
        $$ || schema || $$.$$ || model_table || $$ t1,
        $$ || schema || $$.$$ || scoring_table || $$_imputed t2
) distributed by ($$ || key || $$)$$;

END;
$BODY$
LANGUAGE PLPGSQL;

-----------------------------------------------------------------------------------------------------
-- Sample invocations
-----------------------------------------------------------------------------------------------------
/*
--AT RISK MODEL
select oir.elastic_net_pipeline(
	'oir',
	'at_risk_8_jun_2016',
	'at_risk_input_table_24_may_2016_term',
	'person_uid',
	'first_term_term_gpa',
	ARRAY[ 
	'id',
	'puid', 
	'person_uid',  
	'career_account',
	'first_term_cume_gpa', 
    'at_risk_ind_cume', 
    'first_term_term_gpa',
    'at_risk_ind_term', 
	'profile_academic_period',
	--Remove features which have high correlation with other features 
	'primary_source_ca', 
	'finaid_applicant_ind_n', 
	'app_residency_n',
	'app_race_na'
	], 
	5,
	0.8
	);
*/
