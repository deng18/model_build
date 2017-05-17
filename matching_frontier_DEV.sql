-------------------------------------------------------------
-- Code need testing/debugging!!
-------------------------------------------------------------

drop type if exists oir.matchingFrontier_prep_type cascade;
create type oir.matchingFrontier_prep_type as (id text, matched_id text);

create or replace function oir.matchingFrontier_prep(schema_n_input_table text, id text, outcome_var text, treatment_var text, match_vars text[], distance_measure text, omit_nulls text) 
returns setof oir.matchingFrontier_prep_type as 
$func$
    # Load the package
    library(MatchingFrontier)
    #plan <- pg.spi.prepare(paste('select * from ', schema_n_input_table, sep=''));
    #cursor_obj <- pg.spi.cursor_open('my_cursor', plan);
    #data <- pg.spi.cursor_fetch(cursor_obj, TRUE, as.integer(1000000000));
    sql <- paste('select * from ', schema_n_input_table, sep='')
    data <- pg.spi.exec(sql)

    if (omit_nulls == 'TRUE') {
        data <- na.omit(data)
    } else {
        for(i in 1:ncol(data)){
            data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
        }
    }

    data$key = rownames(data)
    lookup <- data[, c('key',id)]
    data <- sapply(data, function(x) suppressWarnings(as.numeric(as.factor(x))))
    data[, c(treatment_var)] <- as.numeric(as.factor(data[, c(treatment_var)]))-1
    data <- as.data.frame(na.omit(data))

    # Create a vector of column names to indicate which variables we want to match on. 
    # We will match on everything except the id, the treatment and the outcome.
    match.on <- colnames(data)[!(colnames(data) %in% c(id, 'key', outcome_var, treatment_var))]
    # Make the frontier
    if (distance_measure == 'L1') {
        frontier <- makeFrontier(
            dataset = data, 
            treatment = treatment_var, 
            outcome = outcome_var, 
            match.on = match.on,
            QOI = 'SATT',
            metric = 'L1',
            ratio = 'fixed'
            )
    } else if (distance_measure == 'L2') {
        frontier <- makeFrontier(
            dataset = data, 
            treatment = treatment_var, 
            outcome = outcome_var, 
            match.on = match.on
            )
    }

    matched.data <- generateDataset(frontier, N = 10000000000000000000)
    matched.data$key <- rownames(matched.data)

    output <- merge(matched.data, lookup, by = 'key', all.x = TRUE)
    output <- output[colnames(output) %in% c(id, 'matched.to')]
    colnames(output)[colnames(output) == 'matched.to'] <- 'key'
    colnames(lookup)[colnames(lookup) == id] <- 'matched_id'
    output <- merge(output, lookup, by = 'key', all.x = TRUE)
    output <- output[, (colnames(output) %in% c(id, 'matched_id'))]

    #drv <- db.Drive('PostgreSQL')
    #con <- db.Connect(drv)
    #dbWriteTable(con, c(output), value=myTable, overwrite=TRUE, row.names=FALSE)

    return (output)
$func$ language plr;

create or replace function oir.matchingFrontier(schema_n_input_table text, schema_n_output_table text, id text, outcome_var text, treatment_var text, match_vars text[], distance_measure text, omit_nulls boolean) 
returns void as 
$func$
declare t text;

begin
    if omit_nulls = TRUE then t = 'TRUE'; else t = 'FALSE'; end if;

    -- drop view if exists temp_view_to_be_del;
    -- EXECUTE $$create view temp_view_to_be_del as 
    --     (
    --         select t2.*, t1.matched_id
    --         from
    --             (
    --                 select * 
    --                 from oir.matchingFrontier_prep
    --                     (
    --                         '$$ || schema_n_input_table || $$', 
    --                         '$$ || id || $$', 
    --                         '$$ || outcome_var || $$', 
    --                         '$$ || treatment_var || $$', 
    --                         array['$$ || array_to_string(match_vars, $a$', '$a$) || $$'], 
    --                         '$$ || distance_measure || $$',
    --                         '$$ || t || $$'
    --                     )
    --             ) t1
    --         left join 
    --             $$ || schema_n_input_table || $$ t2
    --         on 
    --             t1.$$ || id || $$ = t2.$$ || id || $$
    --     )$$;  

    create table temp_tbl_to_be_del (orig_id text, matched_id text);

    --EXECUTE $$drop table if exists $$ || schema_n_output_table;
    EXECUTE $$--create table $$ || schema_n_output_table || $$ as (
        select * 
        from oir.matchingFrontier_prep(
                '$$ || schema_n_input_table || $$', 
                '$$ || id || $$', 
                '$$ || outcome_var || $$', 
                '$$ || treatment_var || $$', 
                array['$$ || array_to_string(match_vars, $a$', '$a$) || $$'], 
                '$$ || distance_measure || $$',
                '$$ || t || $$'
                --)
        ) into temp_tbl_to_be_del$$;

end;      
$func$ language plpgsql;


--sample invocation
/*
select oir.matchingFrontier(
    'xz.matching_test_input', 
    'xz.test_matching_output', 
    'id',
    'gpa', 
    'owl_ind', 
    array['college', 'gender', 'hs_core_gpa'],
    'L2',
    FALSE
    );
*/
