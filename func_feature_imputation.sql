-------------------------------------------------------------------------------------------------------------------------------
--                                 Feature imputation to handle NULL values                                                  
--                              Created by Srivatsan Ramanujam <sramanujam@pivotal.io>, Sep 2015 
--                                Modified by Xi Zhang <zhan2037@purdue.edu>, March 2016
-------------------------------------------------------------------------------------------------------------------------------

create or replace function oir.cast_boolean_to_float8(table_schema text, table_name text, id_column text) returns void as
$BODY$
DECLARE i record;
BEGIN

EXECUTE $$drop table if exists $$ || table_schema || $$.table_to_be_del cascade$$;
EXECUTE $$create table $$ || table_schema || $$.table_to_be_del as 
(
    select * from $$ || table_schema || $$.$$ || table_name || $$
) distributed by ($$ || id_column || $$)$$;

FOR i in EXECUTE $$SELECT column_name
    from 
        information_schema.columns
    where 
        table_schema = '$$ || table_schema || $$' and 
        table_name = 'table_to_be_del' and
        data_type = 'boolean'$$ 
LOOP
    EXECUTE $$alter table $$ || table_schema || $$.table_to_be_del alter column $$ || i.column_name || $$ type int using $$ || i.column_name || $$::int$$;
    EXECUTE $$alter table $$ || table_schema || $$.table_to_be_del alter column $$ || i.column_name || $$ type float8 using $$ || i.column_name || $$::float8$$;
END LOOP;

END;
$BODY$
LANGUAGE PLPGSQL;

-----------------------------------------------------------------------------------------------------
-- 1) Define UDF for feature imputation
-----------------------------------------------------------------------------------------------------

drop function if exists oir.impute_missing_values(
    text, 
    text, 
    text[], 
    text, 
    text, 
    boolean, 
    text
) cascade;

create or replace function oir.impute_missing_values(
    table_schema text, 
    table_name text, 
    exclude_features text[], 
    id_column text,
    label_column text,
    include_constant_term boolean,
    output_table text
)
returns text
as
$$
    import plpy
    import uuid
    #Obtain unique identifier for temp tables
    UNIQUE_IDENTIFIER = str(uuid.uuid4()).replace('-','_')[:20]

    #If label column is specified, don't include it in the list of values to normalize/impute
    excluded_columns = tuple(exclude_features) if not label_column else tuple(exclude_features + [label_column])

    #0) Convert boolean columns to float8
    sql = """
        select oir.cast_boolean_to_float8('{table_schema}', '{table_name}', '{id_column}')
    """.format(
        table_schema = table_schema,
        table_name = table_name,
        id_column = id_column
    )

    plpy.execute(sql)

    #1) Collect list of column names corresponding to the features.
    sql = """
        select 
            array_agg(column_name order by column_name) as features_names
        from
        (
            select 
                column_name::text
            from 
                information_schema.columns
            where 
                table_schema = '{table_schema}' and 
                table_name = 'table_to_be_del' and
                column_name not in {excluded_columns}
            group by 
                column_name
        )tbl
    """.format(
        table_schema = table_schema,
        excluded_columns = excluded_columns
    )

    results = plpy.execute(sql)
    feature_names = results[0].get('features_names')

    #2) Compute mean & stddev for each column. Collect these in two dicts.
    #Create table to persist values in disk
    mean_stddev_sql_w_creat_tbl = """
    drop table if exists mean_stddev_stats_{UUID} cascade;
    create temp table mean_stddev_stats_{UUID}
    (    
        column_name text,
        avg double precision,
        stddev double precision
    ) distributed randomly;
    """.format(
        UUID = UNIQUE_IDENTIFIER
    )

    plpy.execute(mean_stddev_sql_w_creat_tbl)

    #Insert template for computing mean & stddev of all features.
    mean_stddev_template = """
        insert into mean_stddev_stats_{UUID}
        select 
            '{column_name}' as column_name, 
            avg({column_name}), 
            stddev({column_name})
        from 
            {table_schema}.table_to_be_del
    """
    mean_stddev_template_rpt = []

    for feature in feature_names:
        mean_stddev_template_rpt.append(
                mean_stddev_template.format(
                    column_name = feature, 
                    table_schema = table_schema,
                    UUID = UNIQUE_IDENTIFIER
                )
            ) 

    #Insert values one at a time now
    for insert_stmt in mean_stddev_template_rpt:
        plpy.execute(insert_stmt)

    #3) Prepare SQL to insert constant_term if specified in the input arguments
    constant_term_sql = """
    """
    if(include_constant_term):
        if(label_column):
            constant_term_sql = """
            union all
            select 
                {id_column},
                'constant_term' as feat_name,
                1 as feat,
                1 as feat_normalized,
                {label_column}
            from 
                {table_schema}.table_to_be_del
            """.format(
                id_column = id_column,
                table_schema = table_schema,
                label_column = label_column
            )
        else:
            constant_term_sql = """
            union all
            select 
                {id_column},
                'constant_term' as feat_name,
                1 as feat,
                1 as feat_normalized
            from 
                {table_schema}.table_to_be_del
            """.format(
                id_column = id_column,
                table_schema = table_schema
            )

    #Only use those features who have non-zero variance        
    sql = """
        select
            array_agg(column_name order by column_name) as features_names_nonzero_variance
        from
        (
            select
                column_name
            from
                mean_stddev_stats_{UUID}
            where
                stddev > 0
        )q;
    """.format(
        UUID = UNIQUE_IDENTIFIER
    )                
    feature_names_nonzero_variances = plpy.execute(sql)[0].get('features_names_nonzero_variance')

    #4) Now for every row, subtract the mean and divide by stddev
    sql_imputer_normalizer = """
    """

    #If no label column is supplied as input
    if(not label_column):
        sql_imputer_normalizer = """
        drop table if exists {output_table} cascade;
        create table {output_table}
        as
        (
            select 
                {id_column},
                array_agg(feat_name order by feat_name) as feat_name_vect,
                array_agg(feat order by feat_name) as feat_vect,
                array_agg(feat_normalized order by feat_name) as feat_vect_normalized
            from
            (
                select 
                    {id_column},
                    feat_name,
                    feat,
                    (coalesce(feat,avg) - avg)/stddev as feat_normalized
                from
                (
                    select 
                        t1.{id_column},
                        unnest(t2.feat_name_vect) as feat_name,
                        unnest(t1.feat_vect) as feat,
                        unnest(t2.avg_vect) as avg,
                        unnest(t2.stddev_vect) as stddev
                    from
                    (
                        select 
                            {id_column},
                            ARRAY[{feature_names}] as feat_vect 
                        from 
                            {table_schema}.table_to_be_del
                    )t1,
                    (
                        select 
                            array_agg(column_name order by column_name) as feat_name_vect,
                            array_agg(avg order by column_name) as avg_vect,
                            array_agg(stddev order by column_name) as stddev_vect
                        from 
                            mean_stddev_stats_{UUID}
                        where 
                            column_name not in {excluded_columns} and
                            -- Disregard features with zero variance
                            stddev > 0
                    )t2    
                ) tbl1
                --SQL insert for including the constant term
                {constant_term_sql}
            ) tbl2
            group by 
                {id_column}
        ) distributed by ({id_column});        
        """.format(
            id_column = id_column,
            feature_names = ','.join(feature_names_nonzero_variances),
            table_schema = table_schema,
            output_table = output_table,
            excluded_columns = excluded_columns,
            constant_term_sql = constant_term_sql,
            UUID = UNIQUE_IDENTIFIER
        )
    else:
        sql_imputer_normalizer = """
        drop table if exists {output_table} cascade;
        create table {output_table}
        as
        (
            select 
                {id_column},
                array_agg(feat_name order by feat_name) as feat_name_vect,
                array_agg(feat order by feat_name) as feat_vect,
                array_agg(feat_normalized order by feat_name) as feat_vect_normalized,
                {label_column}
            from
            (
                select 
                    {id_column},
                    feat_name,
                    feat,
                    (coalesce(feat,avg) - avg)/stddev as feat_normalized,
                    {label_column}
                from
            
                (
                    select 
                        t1.{id_column},
                        unnest(t2.feat_name_vect) as feat_name,
                        unnest(t1.feat_vect) as feat,
                        unnest(t2.avg_vect) as avg,
                        unnest(t2.stddev_vect) as stddev,
                        t1.{label_column}
                    from
                    (
                        select 
                            {id_column},
                            ARRAY[{feature_names}] as feat_vect,
                            {label_column} 
                        from 
                            {table_schema}.table_to_be_del
                    )t1,
                    (
                        select 
                            array_agg(column_name order by column_name) as feat_name_vect,
                            array_agg(avg order by column_name) as avg_vect,
                            array_agg(stddev order by column_name) as stddev_vect
                        from 
                            mean_stddev_stats_{UUID}
                        where 
                            column_name not in {excluded_columns} and
                            -- Disregard features with zero variance
                            stddev > 0                            
                    )t2    
                
                ) tbl1
                --SQL insert for including the constant term
                {constant_term_sql}
            ) tbl2
            group by {id_column}, {label_column}
        ) distributed by ({id_column});        
        """.format(
            id_column = id_column,
            feature_names = ','.join(feature_names_nonzero_variances),
            table_schema = table_schema,
            output_table = output_table,
            excluded_columns = excluded_columns,
            label_column = label_column,
            constant_term_sql = constant_term_sql,
            UUID = UNIQUE_IDENTIFIER
        )
    plpy.execute(sql_imputer_normalizer)

    #Drop any temporary tables
    sql  = """
        drop table if exists mean_stddev_stats_{UUID};
        drop table if exists {table_schema}.table_to_be_del;
    """.format(
        UUID = UNIQUE_IDENTIFIER,
        table_schema = table_schema
    )
    plpy.execute(sql)

    return """
        Created normalized features table: {output_table}
    """.format(
        output_table = output_table
    )
$$language plpythonu;

-----------------------------------------------------------------------------------------------------
-- 2) Sample invocations
-----------------------------------------------------------------------------------------------------
/*
select oir.impute_missing_values(
    'rr',
    'input_table_for_modeling_round3_all_num',
    -- Columns to exclude
    ARRAY[
        'person_uid', 
        'class_label',
        'four_class_label',
        'three_class_label',
        'profile_academic_period'
    ],
    'person_uid', -- id column
    'three_class_label', -- label column
    TRUE, -- whether to include a constant term (if running regression)
    'sr.input_table_for_modeling_round3_all_num_imputed' -- output table name
);
*/
