#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 23:07:28 2024

@author: komal
"""

import re
import pandas as pd
import config.regions as rgn
import config.country_mapping as cm


def check_annex1(test_region):
  for region in rgn.annex1 + rgn.annex1_regions:
    if test_region == region.lower():
        return 1
  return 0

def check_non_annex1(test_region):
  for region in rgn.non_annex1 + rgn.non_annex1_regions:
    if test_region == region.lower():
        return 1
  return 0

def check_ldc(test_region):
  for region in rgn.ldc + rgn.ldc_regions:
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

  if region in rgn.split_regions_list:
    parts = region.split(' and ')
    suffix = parts[1].split()[-1]
    region = [parts[0]+ ' ' + suffix, parts[1]]
  else:
    region = [region]

  country_mapping_lower = dict((k.lower(), v.lower()) for k,v in cm.country_mapping.items())

  cleaned_region = []

  for r in region:
    if r in country_mapping_lower.keys():
      cleaned_region.append(country_mapping_lower[r])
    else:
      cleaned_region.append(r)

  return cleaned_region

def get_singular_region(country):
  if country in [c.lower() for c in rgn.ldc]:
    return 'LDC'
  if country in [c.lower() for c in rgn.non_annex1 + rgn.non_annex1_regions]:
    return 'NonAnnex1'
  if country in [c.lower() for c in rgn.annex1]:
    return 'Annex1'
  else:
    return ''

def read_data(file_path):
    return pd.read_csv(file_path)

def write_intermediate_data(df_preprocessed):
    df_preprocessed.to_parquet("data/preprocessed_data.parquet", index=False)

def read_vulnerability_data(file_path):
    vulnerability = pd.read_csv(file_path)
    vulnerability = vulnerability[['Name', '2022']]
    vulnerability['Name'] = vulnerability['Name'].apply(clean_regions)
    vulnerability = vulnerability.explode('Name')
    vulnerability.columns = ['Regions_updated', 'VulnerabilityIndex']
    return vulnerability
    
def read_population_data(file_path):
    population = pd.read_csv(file_path)
    population = population[population['Year'] == 2023][['Entity', 'Population (historical)']]
    population['Entity'] = [country.lower() for country in population['Entity']]
    population.loc[population['Entity'] == 'czechia','Entity'] = 'czech republic'
    population.loc[population['Entity'] == 'hong kong','Entity'] = 'hongkong'
    population.loc[population['Entity'] == 'brunei','Entity'] = 'brunei darussalam'
    population = population[population['Entity'].isin(rgn.region_check_list_lower)]
    population = population[~population['Entity'].isin(rgn.exclude_regions)]
    population.columns = ['Regions_updated', 'Population']
    return population