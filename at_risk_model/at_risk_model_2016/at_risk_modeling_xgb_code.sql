-------------------------------------------------------------------------------------
--                                    AT RISK MODEL
-------------------------------------------------------------------------------------
-- proprocessing
drop table if exists oir.at_risk_input_table_24_may_2016_cume cascade;
create table oir.at_risk_input_table_24_may_2016_cume as
(
    select * from oir.at_risk_input_table_24_may_2016 where at_risk_ind_cume is not null
) distributed by (person_uid);

drop table if exists oir.at_risk_input_table_24_may_2016_term cascade;
create table oir.at_risk_input_table_24_may_2016_term as
(
    select * from oir.at_risk_input_table_24_may_2016 where at_risk_ind_term is not null
) distributed by (person_uid);

-- drop table if exists oir.at_risk_scoring_table_24_may_2016_term cascade;
-- create table oir.at_risk_scoring_table_24_may_2016_term as
-- (
--     select * from oir.at_risk_scoring_table_24_may_2016 where at_risk_ind_term is not null
-- ) distributed by (person_uid);

-- model training
select
    oir.xgboost_grid_search(
        'oir',--training table_schema
        'at_risk_input_table_24_may_2016_term',--training table_name                         --CHANGE HERE FOR CUME GPA MOD
        'person_uid', -- id column
        'at_risk_ind_term', -- class label column                                            --CHANGE HERE FOR CUME GPA MOD
        -- Columns to exclude from features (independent variables)
        ARRAY[
            'puid',
            'person_uid', 
            'first_term_cume_gpa',
            'at_risk_ind_cume',
            'first_term_term_gpa',
            'at_risk_ind_term',
            'profile_academic_period'
        ],
        --XGBoost grid search parameters
        $$
        {
            'learning_rate': [0.05, 0.1, 0.2], #Regularization on weights (eta). For smaller values, increase n_estimators
            'max_depth': [12, 13, 14],#Larger values could lead to overfitting
            'subsample': [0.8, 0.9, 1.0],#introduce randomness in samples picked to prevent overfitting
            'colsample_bytree': [0.8, 0.9, 1.0],#introduce randomness in features picked to prevent overfitting
            'min_child_weight': [1, 2, 4],#larger values will prevent over-fitting
            'n_estimators': [200, 300, 500] #More estimators, lesser variance (better fit on test set)
        }
        $$,
        --Grid search parameters temp table (will be dropped when session ends)
        'xgb_params_temp_tbl',
        --Grid search results table.
        'oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt',                            --CHANGE HERE FOR CUME GPA MOD
        --class weights (set it to empty string '' if you want it to be automatic)
        $$
            {
                0:0.2,
                1:0.8                                                            #IF CHANGE WEIGHTS HERE, CHANGE RESULTING TABLE NAMES ('_80_wt') ACCORDINGLY
            }
        $$,
        0.8
    );

--metrics for positive class (at risk class)
select
    params,
    params_indx,
    substring(metrics from 130 for 8) as pos_fscore,
    substring(metrics from 110 for 8) as pos_precision,
    substring(metrics from 120 for 8) as pos_recall
from oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt                   --CHANGE HERE FOR CUME GPA MOD
order by pos_recall desc;

--view metrics for a selected model
select params, params_indx, metrics
from oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt                         --CHANGE HERE FOR CUME GPA MOD
where params_indx = 304                                                                --USE THE INDEX OF THE BEST MOD

--feature importance for the best model
select unnest(fnames) as fnames, unnest(f_importances) as f_importances 
from oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt                         --CHANGE HERE FOR CUME GPA MOD   
where params_indx = 304;                                                                --USE THE INDEX OF THE BEST MOD


--use the best model to score on F15 cohort
select
    oir.xgboost_mdl_score(
        'oir.at_risk_scoring_table_24_may_2016', -- scoring table                --CHANGE HERE FOR CUME GPA MOD
        'person_uid', -- id column
        'at_risk_ind_term', -- class label column, NULL if unavailable                --CHANGE HERE FOR CUME GPA MOD
        'oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt', -- model table   --CHANGE HERE FOR CUME GPA MOD
        'params_indx = 541', -- model filter, set to 'True' if no filter               --USE THE INDEX OF THE BEST MOD
        'oir.at_risk_grid_search_scoring_results_24_may_2016_term_80_wt' --output table            --CHANGE HERE FOR CUME GPA MOD
    );    

-- test performance metrics by varying classification threshold
select *, avg(pred_prob) over(partition by bin) as avg_pred_prob, avg(at_risk_ind_term) over(partition by bin) as avg_at_risk_ind_term
from
(
    select *, ntile(20) over(order by pred_prob) as bin
    from
    (
        select 
            t1.person_uid, at_risk_ind_term_proba_predicted[2] as pred_prob, at_risk_ind_term
        from 
            oir.at_risk_grid_search_scoring_results_24_may_2016_term_80_wt t1,
            oir.at_risk_scoring_table_24_may_2016_term t2                               --CHANGE HERE FOR CUME GPA MOD
        where 
            substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t2.person_uid
    )t1
)t2
order by pred_prob

-- then copy/paste to S:\IDAP\At Risk Model\metrics_by_threshold.xlsx

------------------------------------------------------------------
--extract the list of at risk students from Fall 16 cohort 
------------------------------------------------------------------
--use the best model to score on F16 cohort (REAL PREDICTION)
select
    oir.xgboost_mdl_score(
        'oir.at_risk_201710_pred_table_06_jun_2016', -- scoring table
        'person_uid', -- id column
        NULL, -- class label column, NULL if unavailable                                 
        'oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt', -- model table           --CHANGE HERE FOR CUME GPA MOD
        'params_indx = 541', -- model filter, set to 'True' if no filter                       --USE THE INDEX OF THE BEST MOD
        'oir.at_risk_201710_pred_results_06_jun_2016_term' -- output table               --CHANGE HERE FOR CUME GPA MOD
    );    

drop table if exists oir.at_risk_201710_list_15_jun_2016_term cascade;               --CHANGE HERE FOR CUME GPA MOD
create table oir.at_risk_201710_list_15_jun_2016_term as                             --CHANGE HERE FOR CUME GPA MOD
(
    select 
        t3.college_desc as college,
        t1.class_label_proba_predicted[2] as predicted_at_risk_probability,
        case when t6.cohort = 'SUMMERSTRT' then 'Y' else 'N' end as summer_start_ind,
        t3.id::text as puid, 
        t4.last_name,
        t4.first_name,
        t4.middle_initial,
        t3.major_desc as major,
        t2.hs_gpa as high_school_gpa,
        t2.hs_core_gpa as high_school_core_gpa,
        t3.residency,
        t4.nation_of_citizenship_desc as nation_of_citizenship,
        t4.gender,
        t4.underrepresented_minority_ind,
        t4.reporting_ethnicity,
        t5.satm,
        t5.satv,
        t5.satw,
        t5.sat_total,
        t5.acte,
        t5.actm,
        t5.actr,
        t5.actew,
        t5.acter,
        t5.act_comp,
        t5.toefl
    from 
        oir.at_risk_201710_pred_results_06_jun_2016_term t1                      --CHANGE HERE FOR CUME GPA MOD
    inner join
        sis.applications t3
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t3.person_uid and
        t3.latest_decision in ('AA','AN') and
        t3.campus = 'PWL' and
        t3.admissions_population in ('B', 'SB') and
        t3.student_level in ('U', 'UG') and
        t3.curriculum_priority_number = 1 and
        t3.academic_period in ('201630', '201710')
    left join  
        oir.at_risk_201710_pred_table_06_jun_2016 t2 
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t2.person_uid
    left join  
        sis.person_detail t4
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t4.person_uid
    left join  
        (
            select 
                q1.person_uid,
                max(CASE test WHEN 'S02' THEN test_score::numeric END) as SATM,
                max(CASE test WHEN 'S01' THEN test_score::numeric END) as SATV,
                max(CASE test WHEN 'S07' THEN test_score::numeric END) as SATW,
                max(CASE test WHEN 'A01' THEN test_score::numeric END) as ACTE,
                max(CASE test WHEN 'A02' THEN test_score::numeric END) as ACTM,
                max(CASE test WHEN 'A05' THEN test_score::numeric END) as ACT_comp,
                max(CASE test WHEN 'A03' THEN test_score::numeric END) as ACTR,
                max(CASE test WHEN 'A07' THEN test_score::numeric END) as ACTEW,
                max(SATTOT) as SAT_total,
                max(ACTER) as ACTER,
                max(CASE test WHEN 'TIBT' THEN test_score::numeric END) as TOEFL
            from sis.test_score q1
            left join
            (
                select
                    person_uid,
                    test_date,
                    sum(case when (test = 'S01' or test = 'S02') then test_score::numeric end) as SATTOT,
                    sum(case when (test = 'A01' or test = 'A03') then test_score::numeric end) as ACTER
                from sis.test_score
                group by 1,2
            ) q2
            on q1.person_uid = q2.person_uid
            group by 1
        ) t5
    on
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t5.person_uid
    left join
        sis.student_cohort t6
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t6.person_uid and
        t6.cohort = 'SUMMERSTRT'
    where
        t1.class_label_predicted = '1'         
    order by 
        t3.college_desc, t1.class_label_proba_predicted[2] desc
) distributed by (puid);

\copy (select * from oir.at_risk_201710_list_15_jun_2016_term) to '/home/users/zhan2037/at_risk_201710_list_15_jun_2016_term.csv' csv header       --CHANGE HERE FOR CUME GPA MOD


------------------------------------------------------------------------------------
-- August 23, 2016
-- Re-generate the at risk list using June model
------------------------------------------------------------------------------------

select select oir.at_risk_201710_pred_table_build('oir', 0);

select
    oir.xgboost_mdl_score(
        'oir.at_risk_201710_pred_table_23_aug_2016', -- scoring table
        'person_uid', -- id column
        NULL, -- class label column, NULL if unavailable                                 
        'oir.at_risk_grid_search_mdl_results_24_may_2016_term_80_wt', -- model table           --CHANGE HERE FOR CUME GPA MOD
        'params_indx = 541', -- model filter, set to 'True' if no filter                       --USE THE INDEX OF THE BEST MOD
        'oir.at_risk_201710_pred_results_23_aug_2016_term' -- output table               --CHANGE HERE FOR CUME GPA MOD
    );    

drop table if exists oir.at_risk_201710_list_23_aug_2016_term cascade;               --CHANGE HERE FOR CUME GPA MOD
create table oir.at_risk_201710_list_23_aug_2016_term as                             --CHANGE HERE FOR CUME GPA MOD
(
    select 
        t3.college_desc as college,
        t1.class_label_proba_predicted[2] as predicted_at_risk_probability,
        case when t6.cohort = 'SUMMERSTRT' then 'Y' else 'N' end as summer_start_ind,
        t3.id::text as puid, 
        t4.last_name,
        t4.first_name,
        t4.middle_initial,
        t3.major_desc as major,
        t2.hs_gpa as high_school_gpa,
        t2.hs_core_gpa as high_school_core_gpa,
        t3.residency,
        t4.nation_of_citizenship_desc as nation_of_citizenship,
        t4.gender,
        t4.underrepresented_minority_ind,
        t4.reporting_ethnicity,
        t5.satm,
        t5.satv,
        t5.satw,
        t5.sat_total,
        t5.acte,
        t5.actm,
        t5.actr,
        t5.actew,
        t5.acter,
        t5.act_comp,
        t5.toefl
    from 
        oir.at_risk_201710_pred_results_23_aug_2016_term t1                      --CHANGE HERE FOR CUME GPA MOD
    inner join
        sis.applications t3
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t3.person_uid and
        t3.latest_decision in ('AA','AN') and
        t3.campus = 'PWL' and
        t3.admissions_population in ('B', 'SB') and
        t3.student_level in ('U', 'UG') and
        t3.curriculum_priority_number = 1 and
        t3.academic_period in ('201630', '201710')
    left join  
        oir.at_risk_201710_pred_table_23_aug_2016 t2 
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t2.person_uid
    left join  
        sis.person_detail t4
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t4.person_uid
    left join  
        (
            select 
                q1.person_uid,
                max(CASE test WHEN 'S02' THEN test_score::numeric END) as SATM,
                max(CASE test WHEN 'S01' THEN test_score::numeric END) as SATV,
                max(CASE test WHEN 'S07' THEN test_score::numeric END) as SATW,
                max(CASE test WHEN 'A01' THEN test_score::numeric END) as ACTE,
                max(CASE test WHEN 'A02' THEN test_score::numeric END) as ACTM,
                max(CASE test WHEN 'A05' THEN test_score::numeric END) as ACT_comp,
                max(CASE test WHEN 'A03' THEN test_score::numeric END) as ACTR,
                max(CASE test WHEN 'A07' THEN test_score::numeric END) as ACTEW,
                max(SATTOT) as SAT_total,
                max(ACTER) as ACTER,
                max(CASE test WHEN 'TIBT' THEN test_score::numeric END) as TOEFL
            from sis.test_score q1
            left join
            (
                select
                    person_uid,
                    test_date,
                    sum(case when (test = 'S01' or test = 'S02') then test_score::numeric end) as SATTOT,
                    sum(case when (test = 'A01' or test = 'A03') then test_score::numeric end) as ACTER
                from sis.test_score
                group by 1,2
            ) q2
            on q1.person_uid = q2.person_uid
            group by 1
        ) t5
    on
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t5.person_uid
    left join
        sis.student_cohort t6
    on 
        substring(t1.person_uid from 1 for position('.' in t1.person_uid)-1) = t6.person_uid and
        t6.cohort = 'SUMMERSTRT'
    where
        t1.class_label_predicted = '1'         
    order by 
        t3.college_desc, t1.class_label_proba_predicted[2] desc
) distributed by (puid);

------------------------------------------------------------------------------------------------
-- get students predicted at risk in Aug but not in Jun (new at risk students since june 15)
-- 225 new at risk students
drop table if exists oir.at_risk_onlist_all;
create table oir.at_risk_onlist_all as
(
    select t1.*
    from oir.at_risk_201710_list_23_aug_2016_term t1
    left join oir.at_risk_201710_list_15_jun_2016_term t2 
    on t1.puid=t2.puid 
    where t2.puid is null
) distributed by (puid);

-- due to new admits
-- 11 distinct students
drop table if exists oir.at_risk_onlist_new_admits;
create table oir.at_risk_onlist_new_admits as
(
    select q1.*
    from 
        sis.applications q1,
        (
            select t1.puid 
            from oir.at_risk_201710_list_23_aug_2016_term t1 
            inner join sis.banner_lookup q on t1.puid=q.puid 
            left join oir.at_risk_201710_pred_table_06_jun_2016 t2 on q.person_uid = t2.person_uid 
            where t2.person_uid is null
        )q2
    where q1.id = q2.puid and academic_period in ('201630','201710')
) distributed by (id);

-- due to data updates
drop table if exists at_risk_temp1;
create temp table at_risk_temp1 as
(
    select q2.*, q1.puid 
    from
        (
            select person_uid, t1.puid
            from oir.at_risk_201710_list_23_aug_2016_term t1
            inner join sis.banner_lookup x on t1.puid=x.puid 
            left join oir.at_risk_201710_list_15_jun_2016_term t2 on t1.puid=t2.puid 
            where t2.puid is null
        )q1
    inner join oir.at_risk_201710_pred_table_06_jun_2016 q2
    on q1.person_uid=q2.person_uid
    inner join oir.at_risk_201710_pred_table_23_aug_2016 q3
    on q1.person_uid=q3.person_uid
)distributed by (puid);

drop table if exists at_risk_temp2;
create temp table at_risk_temp2 as
(
    select q3.*, q1.puid 
    from
        (
            select person_uid, t1.puid
            from oir.at_risk_201710_list_23_aug_2016_term t1
            inner join sis.banner_lookup x on t1.puid=x.puid 
            left join oir.at_risk_201710_list_15_jun_2016_term t2 on t1.puid=t2.puid 
            where t2.puid is null
        )q1
    inner join oir.at_risk_201710_pred_table_06_jun_2016 q2
    on q1.person_uid=q2.person_uid
    inner join oir.at_risk_201710_pred_table_23_aug_2016 q3
    on q1.person_uid=q3.person_uid
)distributed by (puid);

-- if need to compare summary stats, persist the above two tables to oir schema
-- select oir.dataset_compare('oir', 'at_risk_temp1', 'at_risk_temp2', array['puid','person_uid'], 'at_risk_compare1')
-- select * from oir.at_risk_compare1

drop table if exists oir.at_risk_onlist_data_change;
create table oir.at_risk_onlist_data_change as 
(
select *  from (
    select date '06-15-2016' as dt_created, * from at_risk_temp1 
    union
    select date '08-23-2016' as dt_created, * from at_risk_temp2 
    )t order by puid
) distributed by (puid);

--------------------------------------------------------------------------------------------
-- get students predicted at risk in Jun but not in Aug 
-- 430 new at risk students
drop table if exists oir.at_risk_offlist_all;
create table oir.at_risk_offlist_all as
(
    select t1.*
    from oir.at_risk_201710_list_15_jun_2016_term t1
    left join oir.at_risk_201710_list_23_aug_2016_term t2 
    on t1.puid=t2.puid 
    where t2.puid is null
) distributed by (puid);

-- due to cancelled admits
-- 53 distinct students
drop table if exists oir.at_risk_offlist_cancelled_admits;
create table oir.at_risk_offlist_cancelled_admits as
(
    select q1.*
    from 
        sis.applications q1,
        (
            select t1.puid 
            from oir.at_risk_201710_list_15_jun_2016_term t1 
            inner join sis.banner_lookup q on t1.puid=q.puid 
            left join oir.at_risk_201710_pred_table_23_aug_2016 t2 on q.person_uid = t2.person_uid  
            where t2.person_uid is null
        )q2
    where q1.id = q2.puid and academic_period in ('201630','201710')
) distributed by (id);

-- due to data updates
drop table if exists at_risk_temp3;
create temp table at_risk_temp3 as
(
    select q2.*, q1.puid 
    from
        (
            select person_uid, t1.puid
            from oir.at_risk_201710_list_15_jun_2016_term t1
            inner join sis.banner_lookup x on t1.puid=x.puid 
            left join oir.at_risk_201710_list_23_aug_2016_term t2 on t1.puid=t2.puid 
            where t2.puid is null
        )q1
    inner join oir.at_risk_201710_pred_table_06_jun_2016 q2
    on q1.person_uid=q2.person_uid
    inner join oir.at_risk_201710_pred_table_23_aug_2016 q3
    on q1.person_uid=q3.person_uid
)distributed by (puid);

drop table if exists at_risk_temp4;
create temp table at_risk_temp4 as
(
    select q3.*, q1.puid 
    from
        (
            select person_uid, t1.puid
            from oir.at_risk_201710_list_15_jun_2016_term t1
            inner join sis.banner_lookup x on t1.puid=x.puid 
            left join oir.at_risk_201710_list_23_aug_2016_term t2 on t1.puid=t2.puid 
            where t2.puid is null
        )q1
    inner join oir.at_risk_201710_pred_table_06_jun_2016 q2
    on q1.person_uid=q2.person_uid
    inner join oir.at_risk_201710_pred_table_23_aug_2016 q3
    on q1.person_uid=q3.person_uid
)distributed by (puid);

-- if need to compare summary stats, persist the above two tables to oir schema
-- select oir.dataset_compare('oir', 'at_risk_temp3', 'at_risk_temp4', array['puid','person_uid'], 'at_risk_compare2')
-- select * from oir.at_risk_compare2

drop table if exists oir.at_risk_offlist_data_change;
create table oir.at_risk_offlist_data_change as 
(
select *  from (
    select date '06-15-2016' as dt_created, * from at_risk_temp3 
    union
    select date '08-23-2016' as dt_created, * from at_risk_temp4 
    )t order by puid
) distributed by (puid);

\copy (select * from oir.at_risk_201710_list_23_aug_2016_term) to '/tmp/at_risk_201710_list_23_aug_2016_term.csv' csv header
\copy (select * from oir.at_risk_onlist_all) to '/tmp/at_risk_onlist_all.csv' csv header
\copy (select * from oir.at_risk_onlist_new_admits) to '/tmp/at_risk_onlist_new_admits.csv' csv header
\copy (select * from oir.at_risk_onlist_data_change) to '/tmp/at_risk_onlist_data_change.csv' csv header
\copy (select * from oir.at_risk_offlist_all) to '/tmp/at_risk_offlist_all.csv' csv header
\copy (select * from oir.at_risk_offlist_cancelled_admits) to '/tmp/at_risk_offlist_cancelled_admits.csv' csv header
\copy (select * from oir.at_risk_offlist_data_change) to '/tmp/at_risk_offlist_data_change.csv' csv header

--save all above in S:\General\OIR\Student\At Risk Regression\Fall 2016\23_aug_2016\new_at_risk_students_since_jun_15.xlsx

