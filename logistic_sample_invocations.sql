-----------------------------------------------------------------------------------------------------
-- Logistic pipeline sample invocations
-----------------------------------------------------------------------------------------------------
--TESTING
select oir.logistic_regression_pipeline(
	'xz',                    --schema for both input and output tables
	'test_mod_v2',           --model name (unique name is recommened)
	'test_mod_input_table',  --input table name
	'person_uid',            --unique identifier of input table
	'new_class',			 --binary dependent variable
	ARRAY[                   --exclusion list (Ex: non-features, reference levels for one-hot encoded features) 
	'person_uid',       
	'class_label',
	'four_class_label',
	'new_class',
	'profile_academic_period'
	],
	5,                       --number of vif steps (max values is total number of independent variables you have)
	0.8, 				     --percent of training vs test split
	4, 						 --number of folds to split training set for ensembling, it needs to be >= 2
	1.0--,                   --weight for positive class (Ex: 1 means equal weight, greater than 1 means positive class has more weight)
	--TRUE                   --not available yet! if causality check for top 10 features is needed
	);

--GRADE MODEL
create table oir.grade_mod_input_table_26_may_2016_bigclasses as
(select * from oir.grade_mod_input_table_netlog_card_lms_sis_26_may_2016 where course_size>=60 and academic_period in ('201510','201520'))
distributed by (pk1);

select oir.logistic_regression_pipeline( 
	'oir', 
	'grade_mod_26_may_2016_bigclasses', 
	'grade_mod_input_table_26_may_2016_bigclasses', 
	'pk1', 
	'grade_2class', 
	ARRAY[ 
	'id',
	'puid', 
	'person_uid', 
	'career_account', 
	'academic_period', 
	'pk1', 
	'course_identification', 
	'course_gpa', 
	'grade_3class', 
	'grade_3class_desc', 
	'grade_2class', 
	'grade_2class_desc', 
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
	'primary_source_ca', 
	'finaid_applicant_ind_n', 
	'residency_f', 
	'race_na' 
	], 
	10, 
	0.8,  
	5,  
	1.0--,
	--FALSE
	); 

--AT RISK MODEL
--model training
select oir.logistic_regression_pipeline(
	'oir',
	'at_risk_18_may_2016',
	'at_risk_input_table_18_may_2016',
	'person_uid',
	'at_risk_ind',
	ARRAY[ 
	'id',
	'puid', 
	'person_uid',  
	'career_account',
	'first_term_cume_gpa', 
	'at_risk_ind', 
	'profile_academic_period',
	--Remove features which have high correlation with other features 
	'primary_source_ca', 
	'finaid_applicant_ind_n', 
	'app_residency_n',
	'person_detail_gender_f',
	'person_detail_urm_n'
	'person_detail_ethnicity_unknown',
	'admitted_college_a'
	], 
	5,
	0.8, 
	5, 
	1.0--,
	--FALSE
	);
