-------------------------------------------------------------------------------------------------------------------------------
--                                      Build Random Forest models in-database                                               --
-------------------------------------------------------------------------------------------------------------------------------
-- Need testing/debugging

-------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------
--1) UDF to train RF
----------------------------------------------------------------------
drop function if exists oir.__scikit_rf_train_parallel__(
    bytea,
    text[],
    text,
    text,
    text,
    numeric
);

create or replace function oir.__scikit_rf_train_parallel__(
    dframe bytea,
    features_all text[],
    class_label text,
    params text,
    class_weights text,
    train_set_size numeric  
)
returns oir.mdl_gridsearch_train_results_type
as
$$
    import plpy, re 
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix    
    from sklearn.ensemble import RandomForestClassifier
    import numpy
    import cPickle as pickle
    import zlib
    import ast
    from operator import itemgetter
    
    #
    def print_prec_rec_fscore_support(mat, metric_labels, class_labels):
        """
           pretty print precision, recall, fscore & support using pandas dataframe
        """
        tbl = pd.DataFrame(mat, columns=metric_labels)
        tbl['class'] = class_labels
        tbl = tbl[['class']+metric_labels]
        return tbl    
    #1) Load the dataset for model training
    df = pickle.loads(zlib.decompress(dframe))

    #2) Train RF model & return a serialized representation to store in table
    features = filter(lambda x: x in df.columns, features_all)
    X = df[features].as_matrix()
    y = df[class_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_set_size))
    #
    sample_representation = y_train.value_counts()
    total_samples = sum(sample_representation)
    sample_weights = None
    if not class_weights:
        sample_weights = map(
                lambda s: total_samples*1.0/sample_representation[s]
                                /
                sum([total_samples*1.0/sample_representation[c] for c in sample_representation.keys()])
                ,
                y_train
            )
    else:
        #User-supplied class-weights
        class_weights_dict = ast.literal_eval(re.sub("[\\t]","",class_weights).strip())
        sample_weights = map(lambda s: class_weights_dict[s], y_train)
    #Train gradient boosted trees
    p_list = [p.split('=') for p in ast.literal_eval(re.sub("[\\t]","",params).strip())]
    params_dict = dict([(k, ast.literal_eval(v.strip())) for k,v in p_list])
    mdl = RandomForestClassifier(**params_dict)
    #Fit model
    mdl.fit(
        X_train, 
        y_train, 
        class_weight = sample_weights
    )    
    #3) Compute and return model metrics score
    y_pred_train = mdl.predict(X_train)
    y_pred_test = mdl.predict(X_test)
    cmat_train = confusion_matrix(y_train, y_pred_train)
    cmat_test = confusion_matrix(y_test, y_pred_test)
    scores = numpy.array(precision_recall_fscore_support(y_test, y_pred_test)).transpose()
    metric_labels = ['precision', 'recall', 'fscore', 'support']
    model_metrics = print_prec_rec_fscore_support(scores, metric_labels, mdl.classes_)
    importance = mdl.feature_importances_
    fnames_importances = sorted(
        [(features[int(k.replace('f',''))], importance[k]) for k in importance], 
        key=itemgetter(1), 
        reverse=True
    )
    fnames, f_importance_scores = zip(*fnames_importances)
    important_features = pd.DataFrame(fnames_importances)
    #return everything
    return (model_metrics.to_string(), features, pickle.dumps(mdl), params, fnames, f_importance_scores)
$$ language plpythonu;

----------------------------------------------------------------------
--2) RF grid search
----------------------------------------------------------------------
drop function if exists oir.scikit_rf_grid_search(
        text, 
        text, 
        text, 
        text, 
        text[], 
        text, 
        text, 
        text,
        text,
        numeric
);

create or replace function oir.scikit_rf_grid_search(
    features_schema text,
    features_tbl text,
    id_column text,
    class_label text,
    exclude_columns text[],
    params_str text,
    grid_search_params_temp_tbl text,
    grid_search_results_tbl text,
    class_weights text,
    train_set_size numeric
)
returns text
as
$$
    import plpy
    import collections, itertools, ast, re
    #1) Expand the grid-search parameters
    params = ast.literal_eval(re.sub("[\\t]","",params_str).strip())
    def expand_grid(params):
        """
           Expand a dict of parameters into a grid
        """
        import collections, itertools
        #Expand the params to run-grid search
        params_list = []
        for key, val  in params.items():
            #If supplied param is a list of values, expand it out
            if(val and isinstance(val, collections.Iterable)):
                r = ["""{k}={v}""".format(k=key,v=v) for v in val]
            else:
                r = ["""{k}={v}""".format(k=key,v=val)]
            params_list.append(r)
        params_grid = [l for l in itertools.product(*params_list)]
        return params_grid

    params_grid = expand_grid(params)

    #2) Save each parameter list in the grid as a row in a distributed table
    sql = """
        drop table if exists {grid_search_params_temp_tbl};
        create temp table {grid_search_params_temp_tbl}
        (
            params_indx int,
            params text
        ) distributed by (params_indx);
    """.format(grid_search_params_temp_tbl=grid_search_params_temp_tbl)
    plpy.execute(sql)
    sql = """
        insert into {grid_search_params_temp_tbl}
            values ({params_indx}, $X${val}$X$);
    """
    for indx, val in enumerate(params_grid):
        plpy.execute(
            sql.format(
                val=val, 
                params_indx = indx+1, #postgres indices start from 1, so keeping it consistent
                grid_search_params_temp_tbl=grid_search_params_temp_tbl
            )
        )

    #3) Extract feature names from information_schema
    discard_features = exclude_columns + [class_label]
    sql = """
        select
            column_name
        from
            information_schema.columns
        where
            table_schema = '{features_schema}' and
            table_name = '{features_tbl}' and
            column_name not in {exclude_columns}
        group by
            column_name
        order by
            column_name
    """.format(
        features_schema = features_schema,
        features_tbl = features_tbl,
        exclude_columns = str(discard_features).replace('[','(').replace(']',')')
    )
    result = plpy.execute(sql)
    features = [r['column_name'] for r in result]

    #4) Extract features from table and persist as serialized dataframe
    sql = """
        drop table if exists {grid_search_params_temp_tbl}_df;
        create temp table {grid_search_params_temp_tbl}_df
        as
        (
            select
                df,
                generate_series(1, {grid_size}) as params_indx
            from
            (
                select
                    oir.__serialize_pandas_dframe_as_bytea__(
                        '{features_schema}',
                        '{features_tbl}',
                        '{id_column}',
                        '{class_label}',
                        ARRAY[{exclude_columns}]
                    ) as df 
            )q
        ) distributed by (params_indx);
    """.format(
        grid_search_params_temp_tbl = grid_search_params_temp_tbl,
        grid_size = len(params_grid),
        features_schema = features_schema,
        features_tbl = features_tbl,
        id_column = id_column,
        class_label = class_label,
        exclude_columns = str(exclude_columns).replace('[','').replace(']','')
    )
    plpy.execute(sql)

    #5) Invoke RF's train by passing each row from parameter list table. This will run in parallel.
    sql = """
        drop table if exists {grid_search_results_tbl};
        create table {grid_search_results_tbl}
        as
        (
            select
                now() as mdl_train_ts,
                '{features_schema}.{features_tbl}'||'_scikit_rf' as mdl_name,
                (mdl_results).metrics,
                (mdl_results).features,
                (mdl_results).mdl,
                (mdl_results).params,
                (mdl_results).fnames,
                (mdl_results).f_importances,
                params_indx
            from
            (
                select
                    oir.__scikit_rf_train_parallel__(
                        df,
                        ARRAY[
                            {features}
                        ],
                        '{class_label}',
                        params,
                        $CW${class_weights}$CW$,
                        {train_set_size}
                    ) as mdl_results,
                    t1.params_indx
                from 
                    {grid_search_params_temp_tbl} t1,
                    {grid_search_params_temp_tbl}_df t2
                where 
                    t1.params_indx = t2.params_indx
            )q
        ) distributed by (params_indx);    
    """.format(
        grid_search_results_tbl = grid_search_results_tbl,
        grid_search_params_temp_tbl = grid_search_params_temp_tbl,
        features = str(features).replace('[','').replace(']','').replace(',',',\n'),
        features_schema = features_schema,
        features_tbl = features_tbl,
        class_label = class_label,
        class_weights = class_weights,
        train_set_size = train_set_size
    )
    plpy.execute(sql)
    return """Grid search results saved in {tbl}""".format(tbl = grid_search_results_tbl)
$$ language plpythonu;

---------------------------------------------------------------------------------------------------------
--  Sample invocation                                                                                  --
---------------------------------------------------------------------------------------------------------

---------------------
-- Model training
---------------------

select
    oir.scikit_rf_grid_search(
        'sr',--training table_schema
        'features_tbl',--training table_name
        'id', -- id column
        'multi_class_label', -- class label column
        -- Columns to exclude from features (independent variables)
        ARRAY[
            'id', 
            'multi_class_label', 
            'year'
        ],
        --XGBoost grid search parameters
        $$
            {
                'max_depth': [32],#Larger values could lead to overfitting
                'min_samples_split': [17],#introduce randomness in samples picked to prevent overfitting
                'n_estimators':[100] #More estimators, lesser variance (better fit on test set)
            }
        $$,
        --Grid search parameters temp table (will be dropped when session ends)
        'xgb_params_temp_tbl',
        --Grid search results table.
        'xz.rf_grid_search_test',
        --class weights (set it to empty string '' if you want it to be automatic)
        $$
            {
                'class_a':0.25,
                'class_b':0.50,
                'class_c':0.25
            }
        $$,
        0.8
    );