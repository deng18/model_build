--------------------------------------------------------------------------------------------------------------------------
--                                  Template for building modeling dataset                                              --
--                                       Meng Deng <deng18@purdue.edu>                                                  --
--                                                 May 2017                                                             --
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
-- Parameter setting for grade model input table building pipeline
-- feat_sources: enter any combination of sis, lms, card, netlog
-- output_schema: where to save the output table.
-- daily_table_waited_days: for features fed by daily refreshed raw tables, how many days you want to wait from last feature refreshed date
--------------------------------------------------------------------------------------------------------------------------


drop function if exists sandbox.model_build(text[], text[], text[], text, int, text[], text, text, text, text[], int, float8, int, float8, text, text, text, text, numeric, text) ;
create or replace function sandbox.model_build(input_daily_refresh_table text[], --Daily census table
		input_census_refresh_table text[], --Census refresh table
		input_dep_table text[], -- dependent variable  
		output_schema text, --output schema
		daily_table_waited_days int, --table refresh cycle, refresh if selected table is older than 120 days
		model_name text[], --Put either ' ', 'XGBoost', or 'LR'.
		unique_ID text, 
		dependent_variable text, 
		exclusion_list text[], 
	        --Logistic Regression parameters
		vif_steps int, 
		perc_train float8, 
		num_train_folds int, 
		pos_class_weight float8,
		--XGBoost grid search parameters
		params_str text,
		--Grid search parameters temp table (will be dropped when session ends)
		result_table text,
		grid_search_params_temp_tbl text,
		class_weights text,
		train_set_size numeric,
		train_set_split_var text  --variable used to do the test/train split. Leave '' if you desire random
		) returns text as
$BODY$

DECLARE 
last_update_date date;
last_update_term text;
current_date_str text;
current_term text;
table_size text;
table_size_num integer;

built_table_for_modeling text;




selected_feat_tables text[];

feature_list text[];
join_condition text;



BEGIN

RAISE NOTICE 'Starting';
current_date_str = to_char(current_date, 'DD_Mon_YYYY');

select academic_period 
from oir.academic_calendar
where current_date >= start_date and current_date <= end_date
into current_term;


--Step 1: Refresh tables

--Loop through the daily refresh tables, pull the last time they were updated, and update if needed
FOR i IN 1..array_upper(input_daily_refresh_table, 1) LOOP
    select statime::date 
    from 
    (
        select t1.schemaname, t1.relname, t2.statime, row_number() over (partition by relname order by statime desc) 
        from  pg_stat_all_tables t1
        join pg_stat_last_operation t2 
        on t1.relid = t2.objid and
        t1.schemaname = substring(input_daily_refresh_table[i] from '#"%#".%' for '#') and
        t1.relname = substring(input_daily_refresh_table[i] from '%.#"%#"' for '#') and
        t2.staactionname = 'CREATE' and 
        t2.stasubtype = 'TABLE'
    ) b
    where row_number = 1 
    into last_update_date;

    --can be changed to higher refresh frequency if needed
    --make sure update functions share the name of the table exactly!
    IF current_date - last_update_date > daily_table_waited_days THEN 
        execute $$select $$ || input_daily_refresh_table[i] || $$()$$;
    ELSE CONTINUE;
    END IF;
END LOOP;

FOR i IN 1..array_upper(input_census_refresh_table, 1) LOOP
    select statime::date 
    from 
    (
        select t1.schemaname, t1.relname, t2.statime, row_number() over (partition by relname order by statime desc) 
        from  pg_stat_all_tables t1
        join pg_stat_last_operation t2 
        on t1.relid = t2.objid and
        t1.schemaname = substring(input_census_refresh_table[i] from '#"%#".%' for '#') and
        t1.relname = substring(input_census_refresh_table[i] from '%.#"%#"' for '#') and
        t2.staactionname = 'CREATE' and 
        t2.stasubtype = 'TABLE'
    )t
    where row_number = 1 
    into last_update_date;

    select academic_period 
    from oir.academic_calendar
    where last_update_date >= start_date and last_update_date <= end_date
    into last_update_term;

    --if not the same term, refresh
    IF current_term != last_update_term THEN 
        execute $$select $$ || input_census_refresh_table[i] || $$()$$;
    ELSE CONTINUE;
    END IF;
END LOOP;



--After refresh, merge the two tables.
selected_feat_tables = null::text[];
selected_feat_tables = input_daily_refresh_table || input_census_refresh_table;


--Step 2: Create a base table

--temp_table includes everything from the dependent variable table. 
EXECUTE 
$$ drop table if exists temp_table cascade;
create temp table temp_table as
      (select * from $$ || input_dep_table[1] || $$) 
distributed by (person_uid)$$;


FOR i in 1..array_upper(selected_feat_tables, 1) LOOP    
    -- Exclude non-feature tables, why?    
    IF selected_feat_tables[i] not in (
        'features_dependent_variables.grade_mod_dep_var',   
        'features_learning_management.bblearn_student_activity'
    ) THEN

        feature_list = null::text[];
        
        --select the names of all feature names into a list, excluding non-feature variables
        select array_agg(column_name::text) as column_name_arr 
        from information_schema.columns
        where 
            table_schema || '.' || table_name = selected_feat_tables[i] and
            column_name not in (
                'person_uid', 
                'id',
                'puid',
                'career_account',
                'academic_period',  
                'aid_year',
                'course_identification',
                'hs_zip',    --hs_zip in high_school table is not pivoted
                'hs_percentile',    --exclude hs_percentile in high_school table, use profile_high_school_percentile in feats_from_profile_table instead
                'profile_academic_period'   --profile_academic_period in feats_from_profile_table is not pivoted
                ) and    
            column_name !~ E'hs\\_nation\\_.*' and  --hs_nation_... columns in high_school table
            column_name !~ E'hs\\_state\\_(?!in|il)[a-z]+' and  --exclude hs_state_... columns in high_school table, except _in and _il states
            column_name !~ E'app\\_race.+' and  --exclude app_race... in application_demos table
            column_name !~ E'app\\_residency\\_.+' and  --exclude app_residency_... columns in application_demos table
            column_name !~ E'profile\\_highest\\_sat\\_.+' and   --exclude profile_highest_sat_... columns in feats_from_profile_table, use test scores in application_tests table instead
            column_name !~ E'person\\_detail\\_.+'  --exclude person_detail_male and person_detail_urm columns in feats_from_person_detail_table
           
        into feature_list;

--Step 3: Build join condition. 
        --detect which type of join condition is required for this table based on the available data, and execute the join
        --make sure to add 'custom' ifs to the top!
        join_condition = null::text;

        IF selected_feat_tables[i] = 'features_financial_aid.feats_from_fin_aid_table_year_level' THEN
            join_condition = 't1.person_uid = t2.person_uid and substring(t1.academic_period::text from 3 for 2) = substring(t2.aid_year::text from 3 for 2)';
        ELSIF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='person_uid') = TRUE THEN
            IF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='academic_period') = TRUE THEN
                IF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='course_identification') = TRUE THEN
                    join_condition = 't1.person_uid = t2.person_uid and t1.academic_period = t2.academic_period and t1.course_identification = t2.course_identification';
                ELSE 
                    join_condition = 't1.person_uid = t2.person_uid and t1.academic_period = t2.academic_period';
                END IF;
            ELSIF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='course_identification') = TRUE THEN
                join_condition = 't1.person_uid = t2.person_uid and t1.course_identification = t2.course_identification';
            ELSE
                join_condition = 't1.person_uid = t2.person_uid';
            END IF;
        ELSIF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='academic_period') = TRUE THEN
            IF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='course_identification') = TRUE THEN
                join_condition = 't1.academic_period = t2.academic_period and t1.course_identification = t2.course_identification';
            ELSE 
                join_condition = 't1.academic_period = t2.academic_period';
            END IF;
        ELSIF exists (select 1 from information_schema.columns where table_schema || '.' || table_name = selected_feat_tables[i] AND column_name='course_identification') = TRUE THEN
            join_condition = 't1.course_identification = t2.course_identification';
        END IF;

	FOR I IN 1..array_upper(feature_list, 1) LOOP
		
		feature_list[I] = 't2.' || feature_list[I];
	END LOOP; 



        drop table if exists new_temp_table cascade;
        EXECUTE $$create temp table new_temp_table as
        (
            select 
                t1.*,
                $$ || array_to_string(feature_list, $a$, $a$) || $$
            from
                temp_table t1
            left join
                $$ || selected_feat_tables[i] || $$ t2
            on $$ || join_condition || $$
        ) distributed by (person_uid)$$;

        drop table if exists temp_table cascade;
        alter table new_temp_table rename to temp_table;
    END IF;
   
END LOOP;


--Section 4: Output table.

EXECUTE $$drop table if exists $$ || output_schema || $$.$$ || $$input_table_$$ || $$_$$ || current_date_str || $$ cascade$$;
EXECUTE $$create table $$ || output_schema || $$.$$ || $$input_table_$$ || $$_$$ || current_date_str || $$ as
(
    select * from temp_table
) distributed by (person_uid)$$;

--Now the input table for the following modeling stage is ready.
built_table_for_modeling = $$input_table_$$ || $$_$$ || current_date_str;

SELECT pg_size_pretty(pg_total_relation_size(output_schema|| $$.$$ || built_table_for_modeling)) into table_size;
RAISE NOTICE 'table size %', table_size;

table_size_num = cast(split_part(table_size, ' ', 1) as int);

-- call a model function
FOR i in 1..array_upper(model_name, 1) LOOP
   --For empty string
   IF model_name[i] = '' THEN
	RAISE NOTICE 'Move next';
   --call logistic regression pipeline
   ELSEIF model_name[i] = 'LR' THEN
	RAISE NOTICE 'Calling %', model_name[i];
	--Call the logistic regression pipeline.
	PERFORM oir.logistic_regression_pipeline( 
	output_schema, --schema
	result_table, -- output table.
	built_table_for_modeling, --input table
	unique_ID, --id column
	dependent_variable, --Dependent variable
	exclusion_list, --columns to exclude '$$ || array_to_string(exclusion_list, $a$, $a$) || $$'
	vif_steps, --num_steps (max values is total number of independent variables you have). 
	perc_train, --percent of training
	num_train_folds, --# of folds for crossed-valiation?
	pos_class_weight --1 if positive and negative classes are weighted equally. 
	);
   ELSEIF model_name[i] = 'XGBoost' AND table_size_num >= 2000 THEN
	RAISE NOTICE 'Table size over 2000 MB, please partition the data and rerun the model!';
   ELSEIF model_name[i] = 'XGBoost' AND table_size_num < 2000 THEN
	--Need to partition the dataset.
	RAISE NOTICE 'Calling %', model_name[i];
	/*IF pg_size_pretty(pg_total_relation_size('output_schema.built_table_for_modeling')) >= 2000 THEN 
		RETURN 'Table size over 2000 MB, please partition the data and rerun the model!';*/
	--Call the xgboost pipeline.
	PERFORM oir.xgboost_grid_search(
	output_schema, --schema
	built_table_for_modeling, --input table: training table_name
	unique_ID, --id column
	dependent_variable, --Dependent variable, or class label.
	exclusion_list, --columns to exclude '$$ || array_to_string(exclusion_list, $a$, $a$) || $$'
	params_str, --Parameters,
	grid_search_params_temp_tbl, --grid search temp table,
	result_table, -- xgb_results table
	class_weights, --class weights (set it to empty string '' if you want it to be automatic)
	train_set_size, 
	train_set_split_var 
	);
	RAISE NOTICE 'XGBoost is completed!';
   ELSE 
	RAISE NOTICE 'Unrecognized model name';
   END IF;
END LOOP;


RETURN 'Completed!';

END;
$BODY$
LANGUAGE PLPGSQL;
