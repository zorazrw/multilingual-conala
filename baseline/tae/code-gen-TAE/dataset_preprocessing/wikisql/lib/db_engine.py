# BSD 3-Clause License
#
# Copyright (c) 2017, Salesforce Research
# All rights reserved.
# LICENSE file in dataset_preporcessing/wikisql/LICENSE
#####################################################################################
# Code is based on https://github.com/salesforce/WikiSQL/blob/master/lib/dbengine.py
####################################################################################

import records
import re
from babel.numbers import parse_decimal, NumberFormatError
from dataset_preprocessing.wikisql.lib.query import Query

# Jan 3, 2019. Wonseok modify the lib. path


schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


class DBEngine:

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()


    def execute_query(self, table_id, query, iws=False, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, ignore_white_space=iws, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True, ignore_white_space=False):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = Query.agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and isinstance(val, str):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            if ignore_white_space and schema['col{}'.format(col_index)] == 'text':
                where_clause.append('REPLACE( col{}, " ", "") {} :col{}'.format(col_index, Query.cond_ops[op], col_index))
            else:
                where_clause.append('col{} {} :col{}'.format(col_index, Query.cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        out = self.conn.query(query, **where_map)
        return [o.result for o in out]
