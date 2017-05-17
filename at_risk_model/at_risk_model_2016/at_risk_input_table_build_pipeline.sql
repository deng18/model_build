--------------------------------------------------------------------------------------------------------------------------
--                                  Modeling dataset building for at risk model based on cume GPA                       --
--                                       Xi Zhang <zhan2037@purdue.edu>                                                 --
--                                                 April 2016                                                           --
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
-- Parameter setting for at risk model input table building pipeline
-- output_schema: where to save the output table.
-- daily_table_waited_days: for features fed by daily refreshed raw tables, how many days you want to wait from last feature refreshed date
--------------------------------------------------------------------------------------------------------------------------

create or replace function oir.at_risk_input_table_build(output_schema text, daily_table_waited_days int) returns void as
$BODY$

DECLARE 
last_update_date date;
last_update_term text;
current_date_str text;
current_term text;
daily_refresh_tables text[];
census_refresh_tables text[];
all_tables text[];
feature_list text[];
join_condition text;

BEGIN
current_date_str = to_char(current_date, 'DD_Mon_YYYY');

select academic_period 
from oir.academic_calendar
where current_date >= start_date and current_date <= end_date
into current_term;

daily_refresh_tables = array[
    'features_dependent_variables.at_risk_dep_var',    --not a feature table!!
    'features_demographics.application_demos',
    'features_demographics.application_tests',
    'features_demographics.admission_attributes',
    'features_demographics.admission_decision',
    'features_demographics.high_school',
    'features_demographics.high_school_inst_gpa',
    'features_demographics.high_school_core_gpa',
    'features_demographics.high_school_driving_time',
    'features_demographics.high_school_to_regional_rep_dist',
    'features_demographics.acs_2015_hs_zip',
    'features_demographics.decision_count',
    'features_demographics.feats_from_person_detail_2016_at_risk'
    ];

census_refresh_tables = array[
    'features_demographics.feats_from_profile_table'   --only need this table to extract profile_academic_period for filtering purposes
    ];

all_tables = daily_refresh_tables || census_refresh_tables;

--Loop through the daily refresh tables, pull the last time they were updated, and update if needed
FOR i IN 1..array_upper(daily_refresh_tables, 1) LOOP
    select statime::date 
    from 
    (
        select t1.schemaname, t1.relname, t2.statime, row_number() over (partition by relname order by statime desc) 
        from  pg_stat_all_tables t1
        join pg_stat_last_operation t2 
        on t1.relid = t2.objid and
        t1.schemaname = substring(daily_refresh_tables[i] from '#"%#".%' for '#') and
        t1.relname = substring(daily_refresh_tables[i] from '%.#"%#"' for '#') and
        t2.staactionname = 'CREATE' and 
        t2.stasubtype = 'TABLE'
    )t
    where row_number = 1 
    into last_update_date;

    --can be changed to higher refresh frequency if needed
    --make sure update functions share the name of the table exactly!
    IF current_date - last_update_date > daily_table_waited_days THEN 
        execute $$select $$ || daily_refresh_tables[i] || $$()$$;
    ELSE CONTINUE;
    END IF;
END LOOP;

FOR i IN 1..array_upper(census_refresh_tables, 1) LOOP
    select statime::date 
    from 
    (
        select t1.schemaname, t1.relname, t2.statime, row_number() over (partition by relname order by statime desc) 
        from  pg_stat_all_tables t1
        join pg_stat_last_operation t2 
        on t1.relid = t2.objid and
        t1.schemaname = substring(census_refresh_tables[i] from '#"%#".%' for '#') and
        t1.relname = substring(census_refresh_tables[i] from '%.#"%#"' for '#') and
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
        execute $$select $$ || census_refresh_tables[i] || $$()$$;
    ELSE CONTINUE;
    END IF;
END LOOP;

--create a 'base' table
drop table if exists at_risk_temp_table cascade;
create temp table at_risk_temp_table as
(
	select 
        *
    from 
        features_dependent_variables.at_risk_dep_var
) distributed by (person_uid);

FOR i in 1..array_upper(all_tables, 1) LOOP
		
	IF all_tables[i] not in (
		'features_dependent_variables.at_risk_dep_var'
	) THEN

		feature_list = null::text[];
		
        --select the names of all feature names into a list, excluding non-feature variables
		select array_agg(column_name::text) as column_name_arr 
		from information_schema.columns
		where 
		    table_schema || '.' || table_name = all_tables[i] and
		    column_name not in (
		        'person_uid', 
		        'id',
		        'puid',
		        'career_account',
		        'hs_zip') and    --hs_zip in high_school table is not pivoted
            column_name !~ E'hs\\_nation\\_.*' and  --hs_nation_... columns in high_school table
            column_name !~ E'hs\\_state\\_(?!in|il)[a-z]+' and  --exclude hs_state_... columns in high_school table, except _in and _il states
            column_name !~ E'profile\\_(?!academic\\_period).+'  --only include profile_academic_period in feats_from_profile_table, for filtering purposes; it should be excluded for modeling
		into feature_list;

        drop table if exists new_at_risk_temp_table cascade;
		EXECUTE $$create temp table new_at_risk_temp_table as
		(
			select 
				t1.*,
				$$ || array_to_string(feature_list, $a$, $a$) || $$
			from
				at_risk_temp_table t1
			left join
				$$ || all_tables[i] || $$ t2
			on t1.person_uid = t2.person_uid
		) distributed by (person_uid)$$;

		drop table if exists at_risk_temp_table cascade;
		alter table new_at_risk_temp_table rename to at_risk_temp_table;

	END IF;
END LOOP;

EXECUTE $$drop table if exists $$ || output_schema || $$.$$ || $$at_risk_input_table_$$ || current_date_str || $$ cascade$$;
EXECUTE $$create table $$ || output_schema || $$.$$ || $$at_risk_input_table_$$ || current_date_str || $$ as
(
	select * from at_risk_temp_table
    where profile_academic_period::int < 201610
) distributed by (person_uid)$$;

EXECUTE $$drop table if exists $$ || output_schema || $$.$$ || $$at_risk_scoring_table_$$ || current_date_str || $$ cascade$$;
EXECUTE $$create table $$ || output_schema || $$.$$ || $$at_risk_scoring_table_$$ || current_date_str || $$ as
(
    select * from at_risk_temp_table
    where profile_academic_period in ('201610')
) distributed by (person_uid)$$;

END;
$BODY$
LANGUAGE PLPGSQL;

--select oir.at_risk_input_table_build('oir', 30);
