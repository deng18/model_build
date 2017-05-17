-------------------------------------------------------------------------------------------------------------------------------
--                                      Build XGBoost models in-database                                                     --
--                                   Srivatsan Ramanujam<vatsan.cs@utexas.edu>                                               --
--                                       Ian Pytlarz<ipytlarz@purdue.edu>                                                    --
-------------------------------------------------------------------------------------------------------------------------------
--
-----------
-- Note: --
-----------
-- 1) The design of this pipeline uses XGBoost (https://github.com/dmlc/xgboost)
--    A grid-search on model parameters is distributed across all nodes such that each node will build a model for a specific 
--    set of parameters. In this sense, training happens in parallel on all nodes. However we're limited by the maximum 
--    field-size in Greenplum/Postgres which is currently 1 GB. 
-- 2) If your dataset is much larger (> 1 GB), it is strongly recommended that you use MADlib's models so that 
--    training & scoring will happen in-parallel on all nodes.
-------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------
--1) XGBoost parallel grid-search
---------------------------------------------------------------------------------------------------------

--1) Create function to construct a dataframe and serialize it.
drop function if exists oir.__serialize_pandas_dframe_as_bytea__(
    text,
    text,
    text,
    text,
    text[]
);

create or replace function oir.__serialize_pandas_dframe_as_bytea__(
    features_schema text,
    features_tbl text,
    id_column text,
    class_label text,
    exclude_columns text[]
)
returns bytea
as
$$
    import pandas as pd
    import zlib
    import cPickle as pickle
    from sklearn.preprocessing import Imputer    
    #http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn/25562948#25562948
    from sklearn.base import TransformerMixin
    import numpy as np
    class DataFrameImputer(TransformerMixin):
        def __init__(self):
            """Impute missing values.
            Columns of dtype object are imputed with the most frequent value 
            in column.
            Columns of other types are imputed with mean of column.
            """
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)

            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)    
    #1) Extract feature names from information_schema
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
    #2) Fetch dataset for model training
    mdl_train_sql = """
        select
            {id_column},
            {features},
            {class_label}
        from
            {features_schema}.{features_tbl}
    """.format(
        features_schema = features_schema,
        features_tbl = features_tbl,
        features = ','.join(features),
        id_column = id_column,
        class_label = class_label
    )

    result = plpy.execute(mdl_train_sql)
    df = pd.DataFrame.from_records(result)
    #Drop any columns which are all null
    df_filtered = df.dropna(axis=1, how='all')
    #Impute missing values before persisting this DFrame
    #MEAN IMPUTATION IS UN-NECESSARY, AND WOULD NEED TO BE SAVED OFF IF WE USED IT
    #imp = DataFrameImputer()
    #imp.fit(df_filtered)
    #df_imputed = imp.transform(df_filtered)

    #compress and output
    compressed = zlib.compress(pickle.dumps(df_filtered))
    return compressed
$$ language plpythonu;


--2) UDF to train XGBoost
drop type if exists oir.mdl_gridsearch_train_results_type cascade;
create type oir.mdl_gridsearch_train_results_type
as
(
    metrics text,
    features text[],
    mdl bytea,
    params text,
    fnames text[],
    f_importances text[],
    precision text[],
    recall text[],
    fscore text[],
    support text[],
    test_ids text[]
);

drop function if exists oir.__xgboost_train_parallel__(
    bytea,
    text[],
    text,
    text,
    text,
    numeric
);

create or replace function oir.__xgboost_train_parallel__(
    dframe bytea,
    features_all text[],
    class_label text,
    params text,
    class_weights text,
    train_set_size numeric,
    id_column text,
    train_set_split_var text
)
returns oir.mdl_gridsearch_train_results_type
as
$$
    import plpy, re 
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix    
    import xgboost as xgb
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

    #2) Train XGBoost model & return a serialized representation to store in table
    features_all.append(id_column)
    features = filter(lambda x: x in df.columns, features_all)
    X = df[features].as_matrix()
    y = df[class_label]
    if train_set_split_var == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_set_size))    
        #We don't actually want the test set size to change. We want it to be constant as we change train set size so we can compare apples to apples
        #so lets lock it at 20% (only less if the train size is > 80%)
        test_set_size = min((1-train_set_size),0.2)
        X_test = X_test[range(0,int(len(y)*test_set_size)),]
        y_test = y_test.head(int(len(y)*test_set_size))
    else:
        split_indx = numpy.where(features == train_set_split_var)[0]
        X = numpy.delete(X,split_indx,1)
        X_train = X[numpy.array(df[train_set_split_var]==1),]
        X_test = X[numpy.array(df[train_set_split_var]==0),]
        y_train = y[numpy.array(df[train_set_split_var]==1)]
        y_test = y[numpy.array(df[train_set_split_var]==0)]
    #save off and remove the id_column for later output. Make sure to get rid of id_column from features!
    test_ids = X_test [:,len(features)-1]
    X_train = numpy.delete(X_train,len(features)-1,1)
    X_test = numpy.delete(X_test,len(features)-1,1)
    features = features[0:len(features)-1]
    #run weights
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
    gbm = xgb.XGBClassifier(**params_dict)
    #Fit model
    gbm.fit(
        X_train, 
        y_train, 
        eval_metric = 'auc',
        sample_weight = sample_weights
    )    
    #3) Compute and return model metrics score
    y_pred_train = gbm.predict(X_train)
    y_pred_test = gbm.predict(X_test)
    cmat_train = confusion_matrix(y_train, y_pred_train)
    cmat_test = confusion_matrix(y_test, y_pred_test)
    scores = numpy.array(precision_recall_fscore_support(y_test, y_pred_test)).transpose()
    metric_labels = ['precision', 'recall', 'fscore', 'support']
    model_metrics = print_prec_rec_fscore_support(scores, metric_labels, gbm.classes_)
    #4) Calculate feature importance scores
    importance = gbm.booster().get_fscore()
    fnames_importances = sorted(
        [(features[int(k.replace('f',''))], importance[k]) for k in importance], 
        key=itemgetter(1), 
        reverse=True
    )
    fnames, f_importance_scores = zip(*fnames_importances)
    important_features = pd.DataFrame(fnames_importances)
    #return everything
    return (model_metrics.to_string(), features, pickle.dumps(gbm), params, fnames, f_importance_scores, 
        model_metrics.iloc[:,1].values.tolist(), model_metrics.iloc[:,2].values.tolist(),
        model_metrics.iloc[:,3].values.tolist(),model_metrics.iloc[:,4].values.tolist(),
        test_ids)
$$ language plpythonu;


--3) XGBoost grid search
drop function if exists oir.xgboost_grid_search(
        text, 
        text, 
        text, 
        text, 
        text[], 
        text, 
        text, 
        text,
        text,
        numeric,
        text
);

create or replace function oir.xgboost_grid_search(
    features_schema text,
    features_tbl text,
    id_column text,
    class_label text,
    exclude_columns text[],
    params_str text,
    grid_search_params_temp_tbl text,
    grid_search_results_tbl text,
    class_weights text,
    train_set_size numeric,
    train_set_split_var text
)
returns text
as
$$
    import plpy
    #1) Expand the grid-search parameters
    import collections, itertools, ast, re
    #Expand the params to run-grid search
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

    #5) Invoke XGBoost's train by passing each row from parameter list table. This will run in parallel.
    sql = """
        drop table if exists {grid_search_results_tbl};
        create table {grid_search_results_tbl}
        as
        (
            select
                now() as mdl_train_ts,
                '{features_schema}.{features_tbl}'||'_xgboost' as mdl_name,
                (mdl_results).metrics,
                (mdl_results).features,
                (mdl_results).mdl,
                (mdl_results).params,
                (mdl_results).fnames,
                (mdl_results).f_importances,
                (mdl_results).precision,
                (mdl_results).recall,
                (mdl_results).fscore,
                (mdl_results).support,
                (mdl_results).test_ids,
                params_indx
            from
            (
                select
                    oir.__xgboost_train_parallel__(
                        df,
                        ARRAY[
                            {features}
                        ],
                        '{class_label}',
                        params,
                        $CW${class_weights}$CW$,
                        {train_set_size},
                        '{id_column}',
                        '{train_set_split_var}'
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
        train_set_size = train_set_size,
        id_column = id_column,
        train_set_split_var = train_set_split_var
    )
    plpy.execute(sql)
    return """Grid search results saved in {tbl}""".format(tbl = grid_search_results_tbl)
$$ language plpythonu;

---------------------------------------------------------------------------------------------------------
--2) XGBoost : Model Scoring
---------------------------------------------------------------------------------------------------------

drop function if exists oir.xgboost_mdl_score(text, text, text, text, text, text);
create or replace function oir.xgboost_mdl_score(
    scoring_tbl text,
    id_column text,
    class_label text,
    mdl_table text,
    mdl_filters text,
    mdl_output_tbl text
)
returns text
as
$$
    import plpy
    import pandas as pd
    from operator import itemgetter
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import xgboost as xgb
    import numpy
    import cPickle as pickle
    from bisect import bisect_left

    #GPDB lacks python 2.7 and numpy 1.9, which allows an interpolation from np.percentile(). Define a function to do this for us instead
    def takeClosest(myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
           return after
        else:
           return before
    
    # Confusion Matrix
    def print_prec_rec_fscore_support(mat, metric_labels, class_labels):
        """
           pretty print precision, recall, fscore & support using pandas dataframe
        """
        tbl = pd.DataFrame(mat, columns=metric_labels)
        tbl['class'] = class_labels
        tbl = tbl[['class']+metric_labels]
        return tbl

    #1) Load the serialized XGBoost model from the table
    mdl_sql = """
        select
            mdl,
            features
        from
            {mdl_table}
        where
            {mdl_filters}
        """.format(
            mdl_table = mdl_table,
            mdl_filters = mdl_filters
        )
    result = plpy.execute(mdl_sql)
    mdl = result[0]['mdl']
    features = result[0]['features']
    #Train gradient boosted trees
    gbm = pickle.loads(mdl)        

    #2) Fetch features from test dataset for scoring
    mdl_score_sql = ""
    if(class_label):
        mdl_score_sql = """
            select
                {id_column},
                {features},
                {class_label}
            from
                {scoring_tbl}
        """.format(
            scoring_tbl = scoring_tbl,
            id_column = id_column,
            features = ','.join(features),
            class_label = class_label
        )
    else:
        mdl_score_sql = """
            select
                {id_column},
                {features}
            from
                {scoring_tbl}
        """.format(
            scoring_tbl = scoring_tbl,
            id_column = id_column,
            features = ','.join(features)
        )
    result = plpy.execute(mdl_score_sql)
    df = pd.DataFrame.from_records(result)
    X_test = df[features]
    y_test = df[class_label] if class_label else None    
    
    #3) Score the test set
    y_pred_test = gbm.predict(X_test.as_matrix())
    y_pred_proba_test = gbm.predict_proba(X_test.as_matrix())
    if(class_label):
        cmat_test = confusion_matrix(y_test, y_pred_test)
        scores = numpy.array(precision_recall_fscore_support(y_test, y_pred_test)).transpose()
        metric_labels = ['precision', 'recall', 'fscore', 'support']
        model_metrics = print_prec_rec_fscore_support(scores, metric_labels, gbm.classes_)
    else:
        model_metrics = 'NA'
    predicted_class_label = class_label+'_predicted' if class_label else 'class_label_predicted'
    predicted_class_proba_label = class_label+'_proba_predicted' if class_label else 'class_label_proba_predicted'
    pred = pd.Series(y_pred_test, index = X_test.index).to_frame(predicted_class_label)
    num_unique_classes = pd.DataFrame(data=pred[predicted_class_label]).apply(lambda x: len(x.unique()))
    pred_proba = pd.DataFrame(data=y_pred_proba_test, 
        index = X_test.index,
        columns = [range(num_unique_classes)])

    res_df = pd.concat([df[id_column],pred,pred_proba],axis=1).set_index(X_test.index)

    #create a combined column list for all the proba values
    res_df['all_class_probas'] = '{' + res_df[0].map(str)
    for class_col in range(1,num_unique_classes):
        res_df['all_class_probas'] = res_df['all_class_probas'] + ',' + res_df[class_col].map(str)
    res_df['all_class_probas'] = res_df['all_class_probas'] + '}'

    #4) Feature importance scores
    importance = gbm.booster().get_fscore()
    fnames_importances = sorted(
                [(features[int(k.replace('f',''))], importance[k]) for k in importance], 
                key=itemgetter(1), 
                reverse=True
            )
    fnames, f_importance_scores = zip(*fnames_importances)
    ret_dict = res_df.to_dict('records')
    ret_result = (
        (
            r[id_column], 
            r[predicted_class_label], 
            r['all_class_probas']
        )
        for r in ret_dict
    )
    
    #5) Create a ROC Curve if testing on a set with class label
    if (class_label):
        class_list = numpy.unique(y_test)
        roc_auc_scores, fpr, tpr, thresholds = [],[],[],[]
        for classname in class_list:
            roc_auc_scores.append(roc_auc_score(numpy.array(y_test)==classname,y_pred_proba_test[:,classname]))
            t_fpr, t_tpr, t_thresholds = roc_curve(numpy.array(y_test),y_pred_proba_test[:,classname],pos_label=classname)
            fpr.append(t_fpr)
            tpr.append(t_tpr)
            thresholds.append(t_thresholds)
        fpr_df = pd.DataFrame(fpr).transpose()
        tpr_df = pd.DataFrame(tpr).transpose()
        thresholds_df = pd.DataFrame(thresholds).transpose()
    else:
        roc_auc_scores = [0]

    #Create a table to hold the unit-level results
    sql = """
        drop table if exists {mdl_output_tbl};
        create table {mdl_output_tbl}
        (
            {id_column} text,
            {predicted_class_label} text,
            {predicted_class_proba_label} float8[]
        ) distributed by ({id_column});
    """.format(
        mdl_output_tbl = mdl_output_tbl,
        id_column = id_column,
        predicted_class_label = predicted_class_label,
        predicted_class_proba_label = predicted_class_proba_label
    )
    plpy.execute(sql)
    sql = """
        insert into {mdl_output_tbl}
        values {row};
    """
    for row in ret_result:
        plpy.execute(sql.format(mdl_output_tbl = mdl_output_tbl, row = row))

    #Create a table for holding the metrics and feature importances
    sql = """
        drop table if exists {mdl_output_tbl}_metrics;
        create table {mdl_output_tbl}_metrics
        (
            precision text[],
            recall text[],
            fscore text[],
            support text[],
            roc_auc_scores text[],
            feature_names text[],
            feature_importance_scores float8[]
        ) distributed randomly;
    """.format(
        mdl_output_tbl = mdl_output_tbl
    )
    plpy.execute(sql)

    #generate metrics for output
    if(class_label):
        precision = str(model_metrics.iloc[:,1].values.tolist()).replace('[','{').replace(']','}').replace('\'','\"')
        recall = str(model_metrics.iloc[:,2].values.tolist()).replace('[','{').replace(']','}').replace('\'','\"')
        fscore = str(model_metrics.iloc[:,3].values.tolist()).replace('[','{').replace(']','}').replace('\'','\"')
        support = str(model_metrics.iloc[:,4].values.tolist()).replace('[','{').replace(']','}').replace('\'','\"')
        roc_auc_scores = str([round(elem,5) for elem in roc_auc_scores]).replace('[','{').replace(']','}').replace('\'','\"')
    else:
        precision = '{NA}'
        recall = '{NA}'
        fscore = '{NA}'
        support = '{NA}'
        roc_auc_scores = '{NA}'
    
    sql = """
        insert into {mdl_output_tbl}_metrics
        values (
            $X${precision}$X$,
            $X${recall}$X$,
            $X${fscore}$X$,
            $X${support}$X$,
            $X${roc_auc_scores}$X$,
            $X${fnames}$X$, 
            $X${f_importances}$X$
            );
    """.format(
        mdl_output_tbl = mdl_output_tbl,
        precision = precision,
        recall = recall,
        fscore = fscore,
        support = support,
        roc_auc_scores = roc_auc_scores,
        fnames = str(fnames).replace('(','{').replace(')','}').replace('\'','\"'),
        f_importances = str(f_importance_scores).replace('(','{').replace(')','}').replace('\'','\"')
    )
    plpy.execute(sql)

    #If a class label was used, create a third output table for roc curves
    if (class_label):
        #calculate 10% of the data points to save, evenly spacesd. We don't need to wait for 100k+ rows to be written to make a good looking curve
        output_length = 1000#round(len(thresholds_df)*0.1,0)
        numbers = list(range(output_length))
        numbers = [100.0*p/output_length for p in numbers]
        thresh_list = sorted(thresholds_df.iloc[:,0].values.tolist())
        thresh_nums = []
        for x in numbers:
            thresh_nums.append(takeClosest(thresh_list,numpy.percentile(thresh_list,x))) 

        thresh_index = []
        for x in thresh_nums:
            thresh_index.append(thresh_list.index(x))

        sql = """
            drop table if exists {mdl_output_tbl}_roc_curve;
            create table {mdl_output_tbl}_roc_curve
            (
                fpr text[],
                tpr text[],
                thresholds text[]
            ) distributed randomly;
        """.format(
            mdl_output_tbl = mdl_output_tbl
        )
        plpy.execute(sql)
        sql = """
            insert into {mdl_output_tbl}_roc_curve
            values (
                $X${fpr}$X$,
                $X${tpr}$X$,
                $X${thresholds}$X$
            );
        """
        for x in thresh_index:
            plpy.execute(sql.format(mdl_output_tbl = mdl_output_tbl, 
                fpr = str(['%.5f' % round(elem,5) for elem in fpr_df.iloc[x].values.tolist()]).replace('[','{').replace(']','}').replace('\'','\"'), 
                tpr = str(['%.5f' % round(elem,5) for elem in tpr_df.iloc[x].values.tolist()]).replace('[','{').replace(']','}').replace('\'','\"'), 
                thresholds = str(['%.5f' % round(elem,5) for elem in thresholds_df.iloc[x].values.tolist()]).replace('[','{').replace(']','}').replace('\'','\"')
                )
            )

    if (class_label):
        return 'Scoring results written to {mdl_output_tbl}\nModel Results:\n{mdl_metrics}'.format(mdl_output_tbl = mdl_output_tbl, 
            mdl_metrics = str(model_metrics.to_string()))
    else:
        return 'Scoring results written to {mdl_output_tbl}'.format(mdl_output_tbl = mdl_output_tbl)
$$language plpythonu;

---------------------------------------------------------------------------------------------------------
--  Sample invocation                                                                                  --
---------------------------------------------------------------------------------------------------------

---------------------
-- Model training
---------------------

-- select
--     oir.xgboost_grid_search(
--         'grade_mdl',--training table_schema
--         'split_temporal_threesemtrain_22_nov_2016',--training table_name
--         'pk1', -- id column
--         'grade_2class_cfail', -- class label column
--         -- Columns to exclude from features (independent variables)
--         ARRAY[
--             'puid',
--             'person_uid', 
--             'academic_period',
--             'course_reference_number',
--             'pk1',
--             'course_identification',
--             'course_gpa',
--             'grade_3class',
--             'grade_3class_desc',
--             'grade_2class',
--             'grade_2class_desc',
--             'grade_2class_cpass',
--             'grade_2class_cpass_desc',
--             'grade_2class_cfail',
--             'grade_2class_cfail_desc'
--         ],
--         --XGBoost grid search parameters
--         $$
--         {
--             'learning_rate': [0.01], #Regularization on weights (eta). For smaller values, increase n_estimators
--             'max_depth': [9,12,15],#Larger values could lead to overfitting
--             'subsample': [0.85],#introduce randomness in samples picked to prevent overfitting
--             'colsample_bytree': [0.85],#introduce randomness in features picked to prevent overfitting
--             'min_child_weight': [100],#larger values will prevent over-fitting
--             'n_estimators':[1500] #More estimators, lesser variance (better fit on test set)
--         }
--         $$,
--         --Grid search parameters temp table (will be dropped when session ends)
--         'xgb_params_temp_tbl',
--         --Grid search results table.
--         'grade_mdl.grid_search_22_nov_2016_temporal_threesemtrain',
--         --class weights (set it to empty string '' if you want it to be automatic)
--         '',
--         0.8,
--         'train_set' --variable used to do the test/train split. Leave NULL if you desire random
--     );

---------------------
-- Model scoring
---------------------

-- select
--     oir.xgboost_mdl_score(
--         'grade_mdl.scoredata_201710_22_nov_2016', -- scoring table
--         'pk1', -- id column
--         NULL, -- class label column, NULL if unavailable
--         'grade_mdl.grid_search_22_nov_2016_temporal_threesemtrain', -- model table
--         'params_indx = 3', -- model filter, set to 'True' if no filter
--         'grade_mdl.score_results_201710_temporal_trainthreesem'
--     );

---------------------------------------------------------------------------------------------------------
--                                                                                                     --
---------------------------------------------------------------------------------------------------------    
