#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:07:58 2024

@author: komal
"""

import re
import os
import sys
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.country_mapping import country_mapping
from config.regions import *

def check_annex1(test_region):
  for region in annex1+annex1_regions:
    if test_region == region.lower():
        return 1
  return 0

def check_non_annex1(test_region):
  for region in non_annex1+non_annex1_regions:
    if test_region == region.lower():
        return 1
  return 0

def check_ldc(test_region):
  for region in ldc+ldc_regions:
    if test_region == region.lower():
      return 1
  return 0

def check_global(region):
  if region == 'global':
      return 1
  return 0

def check_usa(region):
  if region == 'united states':
    return 1
  return 0

def check_india(region):
  if region == 'india':
      return 1
  return 0

def check_china(region):
  if region == 'china':
      return 1
  return 0

def impute_global(x):
  return list(x) if x != set() else ['Global']

def get_research_region(country):
    regions = []
    if check_annex1(country):
        regions.append('Annex1')
    if check_non_annex1(country):
        regions.append('NonAnnex1')
    if check_ldc(country):
        regions.append('LDC')
    if check_usa(country):
        regions.append('USA')
    if check_india(country):
        regions.append('India')
    if check_china(country):
        regions.append('China')
    if not regions:
        regions.append('Unknown')
    return regions


def clean_regions(region):
  region = region.lower()
  region = region.strip()
  region = region.replace('.', '')
  region = region.replace('\xa0', ' ')
  region = re.sub(r"'s\b", "", region)
  region = re.sub(r"’s\b", "", region)

  if region in split_regions_list:
    parts = region.split(' and ')
    suffix = parts[1].split()[-1]
    region = [parts[0]+ ' ' + suffix, parts[1]]
  else:
    region = [region]

  country_mapping_lower = dict((k.lower(), v.lower()) for k,v in country_mapping.items())

  cleaned_region = []

  for r in region:
    if r in country_mapping_lower.keys():
      cleaned_region.append(country_mapping_lower[r])
    else:
      cleaned_region.append(r)

  return cleaned_region

def get_singular_region(country):
  if country in [c.lower() for c in ldc]:
    return 'LDC'
  if country in [c.lower() for c in non_annex1 + non_annex1_regions]:
    return 'NonAnnex1'
  if country in [c.lower() for c in annex1]:
    return 'Annex1'
  else:
    return ''

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
    
    data['Title_Regions'] = data['Title_Regions'].apply(impute_global)
    data['Abstract_Regions'] = data['Abstract_Regions'].apply(impute_global)
    data['Regions'] = data.apply(lambda row: list(set(row['Title_Regions'] + row['Abstract_Regions'])), axis=1)
    
    exploded_data = data.explode('Regions')
    exploded_data['Regions_updated'] = exploded_data['Regions'].apply(clean_regions)
    exploded_data = exploded_data.explode('Regions_updated')
    exploded_data['row_num'] = range(len(exploded_data))
    exploded_data = exploded_data[exploded_data['Regions_updated'].isin(region_check_list_lower)]
    
    exploded_data['Annex1'] = exploded_data['Regions_updated'].apply(lambda x: check_annex1(x))
    exploded_data['NonAnnex1'] = exploded_data['Regions_updated'].apply(lambda x: check_non_annex1(x))
    exploded_data['LDC'] = exploded_data['Regions_updated'].apply(lambda x: check_ldc(x))
    exploded_data['Global'] = exploded_data['Regions_updated'].apply(lambda x: check_global(x))
    exploded_data['USA'] = exploded_data['Regions_updated'].apply(lambda x: check_usa(x))
    exploded_data['India'] = exploded_data['Regions_updated'].apply(lambda x: check_india(x))
    exploded_data['China'] = exploded_data['Regions_updated'].apply(lambda x: check_china(x))
    exploded_data['ResearchRegion'] = exploded_data['Regions_updated'].apply(get_research_region)
    
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
    
    data = pd.read_csv('data/climate_change_research_preprocessed.csv')
    
    df_preprocessed = preprocess_data(data)
    
    df_preprocessed.to_parquet("data/preprocessed_data.parquet", index=False)
    
    print("Saved Preprocessing CSVs")
    
if __name__ == "__main__":
    main()

