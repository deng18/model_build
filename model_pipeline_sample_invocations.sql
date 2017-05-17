
select sandbox.model_build(array[
    'features_dependent_variables.at_risk_dep_var',    --Daily census table
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
    'features_demographics.feats_from_person_detail_table',
    'features_demographics.admitted_college'
    ], 
    array[
    'features_demographics.feats_from_profile_table' -- census refresh table
    ], 
    array['features_dependent_variables.grade_mod_dep_var'], -- dependent variable 
    'md', --output schema
     120, --table refresh cycle, refresh if selected table is older than 120 days
     array[''], --Model statement, leave it blank if you only want to have the input table built step. 
     --array['XGBoost'], --Model statement, run XGBoost model. Right now it is only set up to run one single model at a time, but it is easy to change it to run multiple models in one time and output to multiple result tables.
     --array['LR'], --Model statement, run Logistic Regression model. 
     'pk1', --Unique key.
     'grade_2class_cfail', --Dependent variable. 
	ARRAY['puid', --Exclusion list to build the feature list. If you are running XGBoost model, make sure to exclude categorical variables. 
	'person_uid',
	'academic_period',
	'course_identification',
	'course_reference_number',
	'grade_value',
	'pk1',
	'glass_2class',
	'grade_2class_desc',
	'grade_2class_cfail',
	'grade_2class_cfail_desc',
	'grade_2class_cpass',
	'grade_2class_cpass_desc',
	'grade_3class',
	'grade_3class_desc',
	'at_risk_ind_cume',
	'at_risk_ind_term',
	'primary_source_web',
	'primary_source_manual',
	'primary_source_ca',
	'finaid_applicant_ind_y',
	'finaid_applicant_ind_n',
	'admin_attr_algs',
	'admin_attr_atfc',
	'admin_attr_amwg',
	'admin_attr_aesl',
	'admin_attr_ambb',
	'admin_attr_awsc',
	'admin_attr_arnc',
	'admin_attr_apvt',
	'admin_attr_apmd',
	'admin_attr_aste',
	'admin_attr_awvb',
	'admin_attr_awsw',
	'admin_attr_amsw',
	'admitted_college_it',
	'admitted_college_el',
	'admitted_college_mt',
	'admitted_college_os',
	'admitted_college_hs',
	'admitted_college_cf',
	'admitted_college_m',
	'admitted_college_nr',
	'admitted_college_p',
	'admitted_college_pi',
	'admitted_college_cg',
	'admitted_college_nd',
	'admitted_college_us',
	'admitted_college_pp',
	'admitted_college_la',
	'admitted_college_s',
	'admitted_college_eu',
	'admitted_college_pc',
	'admitted_college_a',
	'admitted_college_e',
	'admitted_college_f',
	'admitted_college_cm',
	'admitted_college_v',
	'admitted_college_at',
	'admitted_college_hh'
	], 
	--The following parameters are parameters for logistic regression model. It could be converted into similar style as the XGBoost model, but 
	1, --VIF steps
	0.8, --training percentage
	4, --training folds
	1, -- POS class weight 
	--XGBoost grid search parameters
        $$ 
        {
            'learning_rate': [0.05], #Regularization on weights (eta). For smaller values, increase n_estimators
            'max_depth': [5],#Larger values could lead to overfitting
            'subsample': [0.85],#introduce randomness in samples picked to prevent overfitting
            'colsample_bytree': [0.85],#introduce randomness in features picked to prevent overfitting
            'min_child_weight': [200],#larger values will prevent over-fitting
            'n_estimators':[150] #More estimators, lesser variance (better fit on test set)
        }
        $$,
        'xgb_params_temp_tbl', --Grid search parameters temp table (will be dropped when session ends)
        'md.test_out', --Grid search results table.
        '',  --class weights (set it to empty string '' if you want it to be automatic)
        0.8, --Train set size.
        '' --variable used to do the test/train split. Leave NULL if you desire random
     );
