--------------------------------------------------------------------------------------------------------------------------
--                                  Modeling dataset building for course grade model                                    --
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
current_date_str = to_char(current_date, 'DD_mon_YYYY');

select academic_period 
from oir.academic_calendar
where current_date >= start_date and current_date <= end_date
into current_term;

daily_refresh_tables = array[
    'features_dependent_variables.grade_mod_dep_var',    --not a feature table!!
    'features_course_schedule.course_size'
    -- 'features_card_service.board_plan_rectrac_buddies',  --broken?
    -- 'features_card_service.board_plan_tran_dining_usage', --needs to be more relevant
    -- 'features_card_service.board_plan_tran_mfu_plan', --needs to be more relevant
    -- 'features_card_service.door_access', --needs to be more relevant
    -- 'features_card_service.rectrac_logins', --not useful for now
    --'features_card_service.weekly_rectrac_swipes',
    -- 'features_card_service.stored_val_tran_dining_usage', --needs to be more relevant
    -- 'features_card_service.stored_val_tran_laundry_usage', --needs to be more relevant
    -- 'features_course_schedule.class_time_schedule_types', --useful/correct?
    -- 'features_course_schedule.class_hr_of_day', --useful/correct?
    -- 'features_course_schedule.feats_from_reg_audit_table', --useful/correct?
    -- 'features_learning_management.lms_db_msg_posted_asof_4_wk_past', --needs to have different inputs!
    -- 'features_learning_management.lms_gradebook_cnt_1st_view_asof_4_wk_past', --needs to have different inputs!
    -- 'features_learning_management.bblearn_student_activity',    --not a feature table!!
    -- 'features_learning_management.lms_num_sessions_dow',
    -- 'features_learning_management.lms_session_duration',
    -- 'features_learning_management.lms_num_courses_per_week',
    -- 'features_learning_management.lms_num_msg_posted_per_course',
    -- 'features_learning_management.lms_msg_participation_ratio_per_course'
    ];

census_refresh_tables = array[
    'features_demographics.application_tests',
    'features_demographics.high_school',
    'features_demographics.high_school_inst_gpa',
    'features_demographics.high_school_core_gpa',
    'features_demographics.acs_2015_hs_zip',
    'features_demographics.same_high_school',
    'features_grade.course_cdfw_rates',
    'features_grade.person_semester_cdfw_rates',
    'features_grade.in_subject_gpa',
    'features_grade.hs_vs_purdue_gpa_diff',
    'features_grade.repeat_course_ind', --change to repeat course gpa?
    'features_grade.prior_purdue_gpa',
    'features_course_schedule.instructor_difficulty_experience',
    'features_academic_study.num_sems_enrolled',
    'features_academic_study.num_class_min_per_week',
    'features_academic_study.schedule_types',
    'features_academic_study.offering_course'
    --'ip.bblearn_gradebook' --needs to have different inputs!
    --'features_grade.cume_ap_credits_earned', --almost no records ever
    --'features_demographics.feats_from_demos_table', --relies on CENSUS. FIX!
    --'features_academic_study.class_boap_enroll_college', --relies on CENSUS. FIX!
    ];

netlog_feat_tables = array[
 	'features_network.attendance_by_person_course'
 	];

sis_feat_tables = array[
    'features_course_schedule.course_size',
	'features_demographics.application_tests',
    'features_demographics.high_school',
    'features_demographics.high_school_inst_gpa',
    'features_demographics.high_school_core_gpa',
    'features_demographics.acs_2015_hs_zip',
    'features_demographics.same_high_school',
    'features_grade.course_cdfw_rates',
    'features_grade.person_semester_cdfw_rates',
    'features_grade.in_subject_gpa',
    'features_grade.hs_vs_purdue_gpa_diff',
    'features_grade.repeat_course_ind',
    'features_grade.prior_purdue_gpa',
    'features_course_schedule.instructor_difficulty_experience',
    'features_academic_study.num_sems_enrolled',
    'features_academic_study.num_class_min_per_week',
    'features_academic_study.schedule_types',
    'features_academic_study.offering_course',
    'features_grade.important_priors'
    ];

card_feat_tables = array[
	'features_card_service.board_plan_rectrac_buddies',
    'features_card_service.weekly_rectrac_swipes',
    'features_card_service.si_sessions'
	];

lms_feat_tables = array[
	'features_learning_management.lms_db_msg_posted_asof_4_wk_past',
    'features_learning_management.lms_gradebook_cnt_1st_view_asof_4_wk_past',
    'ip.gradebook_features'
    -- 'features_learning_management.lms_num_sessions_dow',
    -- 'features_learning_management.lms_session_duration',
    -- 'features_learning_management.lms_num_courses_per_week',
    -- 'features_learning_management.lms_num_msg_posted_per_course',
    -- 'features_learning_management.lms_msg_participation_ratio_per_course'
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
	select 
        *
    from 
        features_dependent_variables.grade_mod_dep_var
) distributed by (person_uid);

FOR i in 1..array_upper(selected_feat_tables, 1) LOOP
		
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
                'profile_academic_period',   --profile_academic_period in feats_from_profile_table is not pivoted
                --section-specific netlog data
                'bytes_per_min_in_class_lec',
                'bytes_per_tran_in_class_lec',
                'perc_attendance_lec',
                'avg_arrival_prompt_lec',
                'bytes_per_min_in_class_lab',
                'bytes_per_tran_in_class_lab',
                'perc_attendance_lab',
                'avg_arrival_prompt_lab',
                'bytes_per_min_in_class_rec',
                'bytes_per_tran_in_class_rec',
                'perc_attendance_rec',
                'avg_arrival_prompt_rec',
                'bytes_per_min_in_class_sd',
                'bytes_per_tran_in_class_sd',
                'perc_attendance_sd',
                'avg_arrival_prompt_sd',
                'bytes_per_min_in_class_pso',
                'bytes_per_tran_in_class_pso',
                'perc_attendance_pso',
                'avg_arrival_prompt_pso',
                'bytes_per_min_in_class_lbp',
                'bytes_per_tran_in_class_lbp',
                'perc_attendance_lbp',
                'avg_arrival_prompt_lbp',
                --tests that don't matter or don't exist anymore
                'aleks',
                'ap_bio',
                'ap_chin',
                'ap_compgov',
                'ap_compscia',
                'ap_eng_lit',
                'ap_envi',
                'ap_euro',
                'ap_german',
                'ap_hum_geo',
                'ap_japa',
                'ap_physb',
                'ap_physc_elec',
                'ap_physc_mech',
                'ap_psyc',
                'ap_span_lang',
                'ap_span_lit',
                'ap_stat',
                'ap_usgov',
                'ap_world_hist',
                --tiny schedule types that are irrelevant
                'sched_type_complex_',
                'sched_type_complex_dis',
                'sched_type_complex_lab',
                'sched_type_complex_labdis',
                'sched_type_complex_lablbp',
                'sched_type_complex_labrecdis',
                'sched_type_complex_lecdis',
                'sched_type_complex_leclabdis',
                'sched_type_complex_leclablbp',
                'sched_type_complex_leclabpso',
                'sched_type_complex_leclabrec',
                'sched_type_complex_leclabrecpso',
                'sched_type_complex_lecpso',
                'sched_type_complex_lecpsodis',
                'sched_type_complex_lecrecdis',
                'sched_type_complex_lecrecpso',
                'sched_type_complex_lecrecpsodis',
                'sched_type_complex_lecsd',
                'sched_type_complex_recdis',
                'sched_type_complex_sd',
                'earned_points',
                'possible_points'
                ) and    
            column_name !~ E'hs\\_nation\\_.*' and  --hs_nation_... columns in high_school table
            column_name !~ E'hs\\_state\\_(?!in|il)[a-z]+' and  --exclude hs_state_... columns in high_school table, except _in and _il states
            column_name !~ E'app\\_race.+' and  --exclude app_race... in application_demos table
            column_name !~ E'app\\_residency\\_.+' and  --exclude app_residency_... columns in application_demos table
            column_name !~ E'profile\\_highest\\_sat\\_.+' and   --exclude profile_highest_sat_... columns in feats_from_profile_table, use test scores in application_tests table instead
            column_name !~ E'person\\_detail\\_.+'   --exclude person_detail_male and person_detail_urm columns in feats_from_person_detail_table
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

--select oir.grade_mod_input_table_build(array['lms','netlog','card','sis'], 'grade_mdl', 10);
--select oir.grade_mod_input_table_build(array['sis'], 'grade_mdl', 10);
