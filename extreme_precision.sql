
create or replace function ip.extreme_precision(
	schema_table_name_score text, 
	schema_table_name_actual text, 
	unique_id text, 
	class_label text, 
	class_number int, 
	low_end numeric, 
	high_end numeric
	) 
RETURNS TABLE(pred_type text, prec numeric) 
AS
$BODY$
BEGIN

drop table if exists prep;
EXECUTE $$create temp table prep
as
(
	select 
		t1.*,
		t2.$$||class_label||$$,
		1 - abs($$||class_label||$$::int-$$||class_label||$$_predicted::int) as correct,
		(CASE
			WHEN $$||class_label||$$_proba_predicted[$$||(class_number+1)||$$] < $$||low_end||$$ THEN 'low'
			WHEN $$||class_label||$$_proba_predicted[$$||(class_number+1)||$$] > $$||high_end||$$ THEN 'high'
			ELSE 'mid' END) as pred_type
	from 
		$$||schema_table_name_score||$$ t1
	left outer join 
		(select $$||class_label||$$, $$||unique_id||$$ from $$||schema_table_name_actual||$$) t2
	on (t1.$$||unique_id||$$=t2.$$||unique_id||$$)
) distributed by ($$||unique_id||$$)$$;

FOR pred_type, prec IN
	select prep.pred_type, avg(prep.correct) from prep group by prep.pred_type
LOOP
	RETURN NEXT;
END LOOP;
RETURN;

END;
$BODY$
LANGUAGE PLPGSQL;

select * from ip.extreme_precision(
	'grade_mdl.score_results_201610_studentsplit',
	'grade_mdl.split_by_student_12_oct_2016_201610',
	'pk1',
	'grade_2class_cfail',
	0,
	0.2,
	0.8
	);