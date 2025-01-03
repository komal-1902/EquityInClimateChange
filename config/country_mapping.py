#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:02:34 2024

@author: komal
"""

country_mapping = {
    'Alabama': 'United States',
    'Alaska': 'United States',
    'Albuquerque': 'United States',
    'Ann Arbor': 'United States',
    'Anchorage': 'United States',
    'Arizona': 'United States',
    'Arkansas': 'United States',
    'Atlanta': 'United States',
    'Berkeley': 'United States',
    'Boston': 'United States',
    'Boulder': 'United States',
    'Buffalo': 'United States',
    'California': 'United States',
    'Clayton': 'United States',
    'Chicago': 'United States',
    'Colorado': 'United States',
    'Connecticut': 'United States',
    'DC': 'United States',
    'Denver': 'United States',
    'Eastern United States': 'United States',
    'Eastern North Carolina': 'United States',
    'Florida': 'United States',
    'Fort Collins': 'United States',
    'Hawaii': 'United States',
    'Honolulu': 'United States',
    'Houston': 'United States',
    'Huntsville': 'United States',
    'IA': 'United States',
    'Indiana': 'United States',
    'Ithaca': 'United States',
    'Knoxville': 'United States',
    'LA': 'United States',
    'La Crosse': 'United States',
    'Long Island': 'United States',
    'Los Angeles': 'United States',
    'Louisiana': 'United States',
    'Lower Colorado': 'United States',
    'MA': 'United States',
    'Manhattan': 'United States',
    'Maryland': 'United States',
    'Massachusetts': 'United States',
    'Miami': 'United States',
    'Michigan': 'United States',
    'Midwest USA': 'United States',
    'Minnesota': 'United States',
    'Missoula': 'United States',
    'MD': 'United States',
    'Nevada': 'United States',
    'New Haven': 'United States',
    'New Jersey': 'United States',
    'New Mexico': 'United States',
    'New York': 'United States',
    'New York City': 'United States',
    'New York Metropolitan': 'United States',
    'North Carolina': 'United States',
    'North Dakota': 'United States',
    'NC': 'United States',
    'NYC': 'United States',
    'Oklahoma': 'United States',
    'PA': 'United States',
    'Pasadena': 'United States',
    'Pennsylvania': 'United States',
    'Philadelphia': 'United States',
    'Piscataway': 'United States',
    'Portland': 'United States',
    'Providence': 'United States',
    'Santa Barbara': 'United States',
    'Santa Cruz': 'United States',
    'San Diego': 'United States',
    'San Francisco': 'United States',
    'Saint Paul': 'United States',
    'Seattle': 'United States',
    'Southeast United States': 'United States',
    'Southwest USA': 'United States',
    'Syracuse': 'United States',
    'Tampa': 'United States',
    'Tempe': 'United States',
    'Texas': 'United States',
    'The Midwest USA': 'United States',
    'The Midwestern United States': 'United States',
    'The Northeastern United States': 'United States',
    'The Southwest United States': 'United States',
    'The Western United States': 'United States',
    'The United States': 'United States',
    'The United States of America': 'United States',
    'US': 'United States',
    'USA': 'United States',
    'US East Coast': 'United States',
    'US Midwest': 'United States',
    'US West Coast': 'United States',
    'Utah': 'United States',
    'United States of America': 'United States',
    'University Park': 'United States',
    'Urbana': 'United States',
    'VA': 'United States',
    'Vermont': 'United States',
    'Washington': 'United States',
    'Western United States': 'United States',
    'Wilmington': 'United States',

    'Beijing': 'China',
    'Central China': 'China',
    'Chongqing': 'China',
    'East China': 'China',
    'Eastern China': 'China',
    'Fuzhou': 'China',
    'Guangxi': 'China',
    'Guangzhou': 'China',
    'Guizhou': 'China',
    'Hangzhou': 'China',
    'Hebei': 'China',
    'Heilongjiang': 'China',
    # 'Hongkong': 'China',
    # 'Hong Kong': 'China',
    'Hubei': 'China',
    'Hubei Province': 'China',
    'Jinan': 'China',
    'Nanchang': 'China',
    'Nanning': 'China',
    'North China': 'China',
    'Northeast China': 'China',
    'Northern China': 'China',
    'Northern Taiwan': 'China',
    'Northwest China': 'China',
    'Northwestern China': 'China',
    "People Republic of China": 'China',
    "People's Republic of China": 'China',
    "People’s Republic of China": 'China',
    'Qingdao': 'China',
    'Qinghai': 'China',
    'Shanghai': 'China',
    'Shanxi': 'China',
    'Shenyang': 'China',
    'Shenzhen': 'China',
    'Shijiazhuang': 'China',
    'Sichuan': 'China',
    'Sichuan Province': 'China',
    'South China': 'China',
    'Southern China': 'China',
    'Southwest China': 'China',
    # 'Taiwan': 'China',
    'Weihai': 'China',
    'Western China': 'China',
    'Wuhan': 'China',
    'Xiamen': 'China',
    'Yunnan': 'China',

    'Britain': 'United Kingdom',
    'Cambridge': 'United Kingdom',
    'England': 'United Kingdom',
    'Great Britain': 'United Kingdom',
    'Greater London': 'United Kingdom',
    'Republic of Ireland': 'United Kingdom',
    'London': 'United Kingdom',
    'New England': 'United Kingdom',
    'Northern Ireland': 'United Kingdom',
    'Southern England': 'United Kingdom',
    'Scotland': 'United Kingdom',
    'The United Kingdom': 'United Kingdom',
    'UK': 'United Kingdom',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Wales': 'United Kingdom',

    'Ahmedabad': 'India',
    'Bengal': 'India',
    'Central India': 'India',
    'Delhi': 'India',
    'Kerala': 'India',
    'Kolkata': 'India',
    'New Delhi': 'India',
    'North India': 'India',
    'Northeast India': 'India',
    'Northeastern India': 'India',
    'Northern India': 'India',
    'Peninsular India': 'India',
    'Punjab': 'India',
    'South India': 'India',
    'Southern India': 'India',

    'Congo, the democratic republic o': 'Democratic Republic of Congo',
    'Congo, the democratic republic of': 'Democratic Republic of Congo',
    'Dem Rep Congo': 'Democratic Republic of Congo',
    'Dem. Rep. Congo': 'Democratic Republic of Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of Congo',
    'DR Congo': 'Democratic Republic of Congo',
    'DR of Congo': 'Democratic Republic of Congo',
    'The Democratic Republic of Congo': 'Democratic Republic of Congo',
    'The Democratic Republic of the Congo': 'Democratic Republic of Congo',
    'Yangambi': 'Democratic Republic of Congo',

    'Greenland': 'Denmark',
    'East Greenland': 'Denmark',
    'North Greenland': 'Denmark',
    'Northeast Greenland': 'Denmark',
    'Northern Greenland': 'Denmark',
    'Northwest Greenland': 'Denmark',
    'Southeast Greenland': 'Denmark',
    'West Greenland': 'Denmark',

    'Brisbane': 'Australia',
    'East Australia': 'Australia',
    'New South Wales': 'Australia',
    'North Australia': 'Australia',
    'Perth': 'Australia',
    'Sydney': 'Australia',

    "Democratic People Republic of Korea": 'North Korea',
    "Democratic People's Republic of Korea": 'North Korea',
    "Korea, Democratic People Repub": 'North Korea',
    "Korea, Democratic People Republic": 'North Korea',
    "Korea, Democratic People's Repub": 'North Korea',
    "Korea, Democratic People's Republic": 'North Korea',

    'Lao': 'Laos',
    'Lao PDR': 'Laos',
    "Lao People's Democratic Republic": 'Laos',
    "Lao People Democratic Republic": 'Laos',
    'LPDR': 'Laos',

    'Ansan Republic of Korea': 'South Korea',
    'Ansan, Republic of Korea': 'South Korea',
    'Busan': 'South Korea',
    'Daegu': 'South Korea',
    'Gwangju': 'South Korea',
    'Korea, republic of': 'South Korea',
    'Republic of Korea': 'South Korea',
    'Seoul': 'South Korea',

    'Antigua': 'Antigua and Barbuda',
    'Barbuda': 'Antigua and Barbuda',
    'Antigua and Barb': 'Antigua and Barbuda',
    'Antigua and Barb.': 'Antigua and Barbuda',

    'Bosnia': 'Bosnia and Herzegovina',
    'Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia and Herz': 'Bosnia and Herzegovina',
    'Bosnia and Herz.': 'Bosnia and Herzegovina',

    'Alberta': 'Canada',
    'Eastern Canada': 'Canada',
    'Ontario': 'Canada',
    'The Northwest territories of Canada': 'Canada',

    'Amsterdam': 'Netherlands',
    'Hague': 'Netherlands',
    'The Netherlands': 'Netherlands',
    'Utrecht': 'Netherlands',

    'St Vincent': 'Saint Vincent and the Grenadines',
    'St Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'St. Vin. and Gren.': 'Saint Vincent and the Grenadines',
    'St Vin and Gren': 'Saint Vincent and the Grenadines',

    'Central Siberia': 'Serbia',
    'Siberia': 'Serbia',
    'West Siberia': 'Serbia',
    'Western Siberia': 'Serbia',

    'Bratislava': 'Slovakia',
    'Bratislava, Bratislava': 'Slovakia',
    'Slovak Republic': 'Slovakia',
    'Svit': 'Slovakia',
    'Zvolen': 'Slovakia',

    'Caribbean': 'Carribean',
    'Puerto': 'Carribean',
    'Puerto Rico': 'Carribean',

    'African Republic': 'Central African Republic',
    'Bangui': 'Central African Republic',
    'Central African Rep': 'Central African Republic',

    'Czhechia': 'Czech Republic',
    'Czechia': 'Czech Republic',
    'The Czech Republic': 'Czech Republic',

    'Dominican Rep': 'Dominican Republic',
    'Dominican Rep.': 'Dominican Republic',

    'INRA': 'France',
    'INRAE': 'France',
    'Paris': 'France',

    'Hateruma': 'Japan',
    'South Japan': 'Japan',
    'Tokyo': 'Japan',

    'Cancun': 'Mexico',
    'México': 'Mexico',
    'Oaxaca': 'Mexico',

    'Ancón': 'Peru',
    'Lima': 'Peru',
    'Perú': 'Peru',

    'Russian Federation': 'Russia',
    'The Russian Federation': 'Russia',
    'Yakutia': 'Russia',

    'Abu Dhabi': 'United Arab Emirates',
    'Dubai': 'United Arab Emirates',
    'UAE': 'United Arab Emirates',

    'Bolivia, Plurinational State of': 'Bolivia',
    'Bolivia (Plurinational State of)': 'Bolivia',

    'São Paulo': 'Brazil',
    'Sao Paulo': 'Brazil',

    'Brunei': 'Brunei Darussalam',
    'Gadong': 'Brunei Darussalam',

    "Republic of Congo": 'Congo',
    'Republic of the Congo': 'Congo',

    'Eq Guinea': 'Equatorial Guinea',
    'Eq. Guinea': 'Equatorial Guinea',

    'Iran, Islamic Republic of': 'Iran',
    'Iran (Islamic Republic of)': 'Iran',

    'Northern Italy': 'Italy',
    'Venice': 'Italy',

    "Côte d'Ivoire": 'Ivory Coast',
    "Cote d'Ivoire": 'Ivory Coast',

    'Marshall Is': 'Marshall Islands',
    'Marshall Is.': 'Marshall Islands',

    'Micronesia (Federated States of)': 'Micronesia',
    'Micronesia, Federated States Of': 'Micronesia',

    'Republic of Moldova': 'Moldova',
    'Moldova, Republic of': 'Moldova',

    'Panamá': 'Panama',
    'Republic of Panama': 'Panama',

    'St Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',

    'Solomon Is': 'Solomon Islands',
    'Solomon Is.': 'Solomon Islands',

    'United Republic of Tanzania': 'Tanzania',
    'Tanzania, United Republic of': 'Tanzania',

    'Trinidad': 'Trinidad and Tobago',
    'Tobago': 'Trinidad and Tobago',

    'Venezuela, Bolivarian Republic o': 'Venezuela',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',

    'Ho Chi Minh City': 'Vietnam',
    'Viet Nam': 'Vietnam',

    "Auckland, New Zealand": 'New Zealand',
    'Bangkok': 'Thailand',
    'Benguela': 'Angola',
    'Cape Town': 'South Africa',
    'Cabo Verde': 'Cape Verde',
    'Dakar': 'Senegal',
    'Dhaka': 'Bangaldesh',
    'Doha': 'Qatar',
    'East Malaysia': 'Malaysia',
    'Guinea Bissau': 'Guinea-Bissau',
    'Hong Kong': 'HongKong',
    'Instituto Geofísico': 'Ecuador',
    'Libreville': 'Gabon',
    'Libyan Arab Jamahiriya': 'Libya',
    'Macedonia': 'North Macedonia',
    'Maradi': 'Niger',
    'Mombasa': 'Kenya',
    'Nicosia': 'Cyprus',
    'Pacific Island': 'Pacific Islands',
    'S Sudan': 'South Sudan',
    'Santa Teresa': 'Costa Rica',
    "São Tomé and Principe": 'Sao Tome and Principe',
    'SIDS': 'Small Island Developing States',
    'Srilanka': 'Sri Lanka',
    'St Lucia': 'Saint Lucia',
    'State of Palestine': 'Palestine',
    'Stockholm': 'Sweden',
    'Sub Saharan Africa': 'Sub-Saharan Africa',
    'Sumatra': 'Indonesia',
    'Swaziland': 'Eswatini',
    'Syrian Arab Republic': 'Syria',
    'Türkiye': 'Turkey',
    'Timor Leste': 'Timor-Leste',
    'Western Norway': 'Norway',
    'Western Uganda': 'Uganda',
    'Yanbu':' Saudi Arabia',
    'Yangon': 'Myanmar',

    'Eastern Asia': 'East Asia',

    'Western Asia': 'West Asia',

    'Southern Asia': 'South Asia',
    'West South Asia': 'South Asia',

    'Equatorial Asia': 'Asia',

    'Mainland Southeast Asia': 'Southeast Asia',
    'South/Southeast asia': 'Southeast Asia',

    'Americas': 'America',
    'Sahel': 'Sahelian Africa',

    'Central North America': 'North America',
    'East North America': 'North America',
    'North-America': 'North America',
    'Northern America': 'North America',
    'Northeast North America': 'North America',
    'Northwestern North America': 'North America',
    'The Northwestern North America': 'North America',
    'Western North America': 'North America',

    'Northeast South America': 'South America',
    'Eastern South America': 'South America',
    'Southeast South America': 'South America',
    'Southeastern South America': 'South America',
    'Southwestern South America': 'South America',

    'Central-East Africa': 'East Africa',
    'Eastern Africa': 'East Africa',
    'Southeastern Africa': 'East Africa',

    'Northern Africa': 'North Africa',

    'Eastern Europe': 'East Europe',
    'South-eastern Europe': 'East Europe',

    'Mediterranean Europe': 'Europe',

    'Northeastern Europe': 'North Europe',
    'Northern Europe': 'North Europe',
    'Northwestern Europe': 'North Europe',
    'Northwest Europe': 'North Europe',

    'Western Europe': 'West Europe',

    'Southern Europe': 'South Europe',
    }