--------------------------------------------------------------------------------------------------------------------------
--                                  Template for building modeling dataset                                              --
--                                       Xi Zhang <zhan2037@purdue.edu>                                                 --
--                                                 April 2016                                                           --
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
-- Parameter setting for grade model input table building pipeline
-- feat_sources: enter any combination of sis, lms, card, netlog
-- output_schema: where to save the output table.
-- daily_table_waited_days: for features fed by daily refreshed raw tables, how many days you want to wait from last feature refreshed date
--------------------------------------------------------------------------------------------------------------------------

create or replace function oir.grade_mod_input_table_build(feat_sources text[], output_schema text, daily_table_waited_days int) returns void as
$BODY$

DECLARE 
last_update_date date;
last_update_term text;
current_date_str text;
current_term text;
daily_refresh_tables text[];
census_refresh_tables text[];
netlog_feat_tables text[];
sis_feat_tables text[];
lms_feat_tables text[];
card_feat_tables text[];
selected_feat_tables text[];
feature_list text[];
join_condition text;

BEGIN
current_date_str = to_char(current_date, 'DD_Mon_YYYY');

select academic_period 
from oir.academic_calendar
where current_date >= start_date and current_date <= end_date
into current_term;

-- Group tables for refresh purpose
daily_refresh_tables = array[
    'features_dependent_variables.grade_mod_dep_var',    --not a feature table!!
    'features_learning_management.bblearn_student_activity',    --not a feature table!!
    'features_learning_management.lms_num_sessions_dow',
    ...
    ];

census_refresh_tables = array[
    'features_demographics.feats_from_profile_table',
    ...
    ];

-- Group tables based on their sources
netlog_feat_tables = array[
    ...
    ];

sis_feat_tables = array[
    ...
    ];

card_feat_tables = array[
    ...
    ];

lms_feat_tables = array[
    ...
    ];

--Combine selected feature tables
selected_feat_tables = null::text[];
FOR i in 1..array_upper(feat_sources, 1) LOOP
    if feat_sources[i] = 'sis' then selected_feat_tables = selected_feat_tables || sis_feat_tables;
    elseif feat_sources[i] = 'lms' then selected_feat_tables = selected_feat_tables || lms_feat_tables;
    elseif feat_sources[i] = 'card' then selected_feat_tables = selected_feat_tables || card_feat_tables;
    elseif feat_sources[i] = 'netlog' then selected_feat_tables = selected_feat_tables || netlog_feat_tables;
    end if;
end loop;

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
drop table if exists grade_mod_temp_table cascade;
create temp table grade_mod_temp_table as
(
    select * from features_dependent_variables.grade_mod_dep_var
) distributed by (person_uid);

FOR i in 1..array_upper(selected_feat_tables, 1) LOOP    
    -- Exclude non-feature tables    
    IF selected_feat_tables[i] not in (
        'features_dependent_variables.grade_mod_dep_var',   
        'features_learning_management.bblearn_student_activity',
        ...
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
                'profile_academic_period',   --profile_academic_period in feats_from_profile_table is not pivoted
                ...
                ) and    
            column_name !~ E'hs\\_nation\\_.*' and  --hs_nation_... columns in high_school table
            column_name !~ E'hs\\_state\\_(?!in|il)[a-z]+' and  --exclude hs_state_... columns in high_school table, except _in and _il states
            column_name !~ E'app\\_race.+' and  --exclude app_race... in application_demos table
            column_name !~ E'app\\_residency\\_.+' and  --exclude app_residency_... columns in application_demos table
            column_name !~ E'profile\\_highest\\_sat\\_.+' and   --exclude profile_highest_sat_... columns in feats_from_profile_table, use test scores in application_tests table instead
            column_name !~ E'person\\_detail\\_.+' and  --exclude person_detail_male and person_detail_urm columns in feats_from_person_detail_table
            ...
        into feature_list;

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

        drop table if exists new_grade_mod_temp_table cascade;
        EXECUTE $$create temp table new_grade_mod_temp_table as
        (
            select 
                t1.*,
                $$ || array_to_string(feature_list, $a$, $a$) || $$
            from
                grade_mod_temp_table t1
            left join
                $$ || selected_feat_tables[i] || $$ t2
            on $$ || join_condition || $$
        ) distributed by (person_uid)$$;

        drop table if exists grade_mod_temp_table cascade;
        alter table new_grade_mod_temp_table rename to grade_mod_temp_table;

    END IF;
END LOOP;

EXECUTE $$drop table if exists $$ || output_schema || $$.$$ || $$grade_mod_input_table_$$ || array_to_string(feat_sources, $$_$$) || $$_$$ || current_date_str || $$ cascade$$;
EXECUTE $$create table $$ || output_schema || $$.$$ || $$grade_mod_input_table_$$ || array_to_string(feat_sources, $$_$$) || $$_$$ || current_date_str || $$ as
(
    select * from grade_mod_temp_table
) distributed by (person_uid)$$;

END;
$BODY$
LANGUAGE PLPGSQL;

--select oir.grade_mod_input_table_build(array['lms','sis'], 'grade_mdl', 3);
