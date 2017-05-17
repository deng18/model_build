create or replace function ip.split_train_test(schema_table_in text, schema_table_out text, variable text) returns void as
$BODY$
BEGIN

drop table if exists count_by;
EXECUTE $$create temp table count_by
as
(
	select
		q.*,
		sem_count,
		sum(count) OVER (PARTITION BY q.academic_period ORDER BY rnum) as cume_sum
	from
	(
		select 
			academic_period, 
			$$ || variable || $$,
			count(*) as count,
			row_number() OVER (PARTITION BY academic_period) as rnum
		from $$ || schema_table_in || $$
		group by 1,2
	)q
	left outer join
	(
		select 
			academic_period, 
			count(*) as sem_count
		from $$ || schema_table_in || $$
		group by 1
	)q2
	on (q.academic_period=q2.academic_period)
) distributed randomly$$;

EXECUTE $$drop table if exists $$||schema_table_out||$$ cascade$$;
EXECUTE $$create table $$||schema_table_out||$$
as
(
	select 
		t.*,
		(CASE WHEN cume_sum < sem_count * 0.8 THEN 1 ELSE 0 END) as train_set
	from 
		$$ || schema_table_in || $$ t
	left outer join
		count_by tt
	on (t.$$||variable||$$=tt.$$||variable||$$ and t.academic_period=tt.academic_period)
) distributed by (person_uid)$$;

END;
$BODY$
LANGUAGE PLPGSQL;

select ip.split_train_test('grade_mdl.grade_mod_input_table_sis_28_Oct_2016','grade_mdl.split_by_course_28_oct_2016','person_uid');


-- --test person splits
-- drop table if exists split_analysis;
-- create temp table split_analysis
-- as
-- (
-- 	select 
-- 		train.course_identification,
-- 		train.academic_period,
-- 		train_cnt,
-- 		test_cnt
-- 	from
-- 		(
-- 			select 
-- 				course_identification, 
-- 				academic_period,
-- 				count(*) as train_cnt
-- 			from 
-- 				data
-- 			where train_set = 1
-- 			group by 1,2
-- 		) train
-- 	inner join
-- 		(
-- 			select 
-- 				course_identification, 
-- 				academic_period,
-- 				count(*) as test_cnt
-- 			from 
-- 				data
-- 			where train_set = 0
-- 			group by 1,2
-- 		) test
-- 	on (train.course_identification=test.course_identification and train.academic_period=test.academic_period)
-- ) distributed by (course_identification);
