#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:07:58 2024

@author: komal
"""

import os
import sys
import ast
import pandas as pd

import helper_functions as func
import regions as rgn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def preprocess_data(data):
    data['Title_Regions'] = data['Title_Regions'].apply(ast.literal_eval)
    data['Abstract_Regions'] = data['Abstract_Regions'].apply(ast.literal_eval)
    data['Date Modified'] = data['Date Modified'].fillna(data['Date Published'])
    data['Year Published'] = pd.to_datetime(data['Date Published']).dt.year.astype(str)
    data['Keywords'] = data['Keywords'].fillna('[]')
    data['Accesses'] = data['Accesses'].fillna(0)
    data['Altmetric'] = data['Altmetric'].fillna(0)
    data['Citation'] = data['Citation'].fillna(0)
    data['Downloads'] = data['Downloads'].fillna(0)
    data['Other Metrics'] = data['Other Metrics'].fillna('')
    data['Text'] = data['Text'].fillna('')
    data['Methods'] = data['Methods'].fillna('')
    data['Conclusions'] = data['Conclusions'].fillna('')
    
    data['Title_Regions'] = data['Title_Regions'].apply(func.impute_global)
    data['Abstract_Regions'] = data['Abstract_Regions'].apply(func.impute_global)
    data['Regions'] = data.apply(lambda row: list(set(row['Title_Regions'] + row['Abstract_Regions'])), axis=1)
    
    exploded_data = data.explode('Regions')
    exploded_data['Regions_updated'] = exploded_data['Regions'].apply(func.clean_regions)
    exploded_data = exploded_data.explode('Regions_updated')
    exploded_data['row_num'] = range(len(exploded_data))
    exploded_data = exploded_data[exploded_data['Regions_updated'].isin(rgn.region_check_list_lower)]
    
    exploded_data['Annex1'] = exploded_data['Regions_updated'].apply(lambda x: func.check_annex1(x))
    exploded_data['NonAnnex1'] = exploded_data['Regions_updated'].apply(lambda x: func.check_non_annex1(x))
    exploded_data['LDC'] = exploded_data['Regions_updated'].apply(lambda x: func.check_ldc(x))
    exploded_data['Global'] = exploded_data['Regions_updated'].apply(lambda x: func.check_global(x))
    exploded_data['USA'] = exploded_data['Regions_updated'].apply(lambda x: func.check_usa(x))
    exploded_data['India'] = exploded_data['Regions_updated'].apply(lambda x: func.check_india(x))
    exploded_data['China'] = exploded_data['Regions_updated'].apply(lambda x: func.check_china(x))
    exploded_data['ResearchRegion'] = exploded_data['Regions_updated'].apply(func.get_research_region)
    
    data_subset = exploded_data[['Title', 'Description', 'Date Published', 'Year Published', 'Date Modified', 'Keywords',
                                 'Journal', 'DOI', 'Authors', 'CorrespondingAuthors', 'AuthorCountry',
                                 'Accesses', 'Altmetric', 'Citation', 'Downloads', 'Other Metrics',
                                 'Abstract', 'Text', 'Methods', 'Conclusions', 'Regions_updated', 'ResearchRegion']]
    data_subset = data_subset[data_subset['ResearchRegion'].apply(lambda x: 'Unknown' not in x)]
    data_subset['ResearchRegion'] = data_subset["ResearchRegion"].astype(str)
    
    group_by_columns = list(data_subset.columns[data_subset.columns != 'Regions_updated'])
    data_grouped = data_subset.groupby(group_by_columns, as_index=False).agg({'Regions_updated': lambda x: list(set(list(x)))})
    data_grouped['ResearchRegion'] = data_grouped['ResearchRegion'].apply(ast.literal_eval)

    return data_grouped
    
    
def main():
    
    data = func.read_data('data/climate_change_research_preprocessed.csv')
    
    df_preprocessed = preprocess_data(data)
    
    func.write_intermediate_data(df_preprocessed)
    
    print("Saved Preprocessing CSVs")
    
if __name__ == "__main__":
    main()

