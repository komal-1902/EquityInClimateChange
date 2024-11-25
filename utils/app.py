#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:33:51 2024

@author: komal
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.country_mapping import country_mapping
from config.regions import *
from preprocessing import *

@st.cache_data
def load_data(file_path):
    """Load preprocessed data."""
    return pd.read_parquet(file_path)

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
    population = population[population['Entity'].isin(region_check_list_lower)]
    population = population[~population['Entity'].isin(exclude_regions)]
    population.columns = ['Regions_updated', 'Population']
    return population

def prepare_graph_data(data_grouped, vulnerability, population):
    data_country = data_grouped[['Regions_updated', 'ResearchRegion', 'DOI', 'Altmetric']]
    data_country = data_country.explode('Regions_updated')
    data_country = data_country[~data_country['Regions_updated'].isin(exclude_regions)]
    
    data_country = pd.merge(data_country, vulnerability, on='Regions_updated', how='left')
    data_country = pd.merge(data_country, population, on='Regions_updated', how='left')
    data_country['VulnerabilityIndex'] = data_country['VulnerabilityIndex'].fillna(0)
    
    data_country['ResearchRegion'] = data_country['ResearchRegion'].astype(str)
    data_country_grouped = data_country.groupby(['Regions_updated', 'ResearchRegion', 'VulnerabilityIndex', 'Population'])\
                                       .agg({'DOI': 'count', 'Altmetric': 'sum'}).reset_index()
    data_country_grouped.columns = ['Regions_updated', 'ResearchRegion', 'VulnerabilityIndex', 'Population', 'NumPublications', 'Altmetrics']
    
    data_country_grouped['NumPublications_log'] = np.log(data_country_grouped['NumPublications'])
    data_country_grouped['NumPublications_sqrt'] = np.sqrt(data_country_grouped['NumPublications'])
    data_country_grouped['Altmetrics_log'] = np.log(data_country_grouped['Altmetrics'])
    data_country_grouped['Altmetrics_sqrt'] = np.sqrt(data_country_grouped['Altmetrics'])
    
    data_country_grouped['ResearchRegion'] = data_country_grouped['ResearchRegion'].apply(ast.literal_eval)
    data_country_grouped = data_country_grouped.explode('ResearchRegion')
    
    return data_country_grouped

def plot_bar(data_grouped, col, agg_func, col_name):
    # Create the data
    categories = ['Annex1', 'NonAnnex1', 'LDC', 'USA', 'China', 'India']
    
    if col == 'Author':
        
        data_grouped['AuthorCountry'] = data_grouped['AuthorCountry'].apply(ast.literal_eval)
        
        annex1_count = 0
        non_annex1_count = 0
        ldc_count = 0
        usa_count = 0
        india_count = 0
        china_count = 0
        
        for author_dict in data_grouped['AuthorCountry']:
          for author, country in author_dict.items():
            if country:
              country = clean_regions(country)[0]
              region = get_research_region(country)
              if 'Annex1' in region:
                annex1_count += 1
              if 'NonAnnex1' in region:
                non_annex1_count += 1
              if 'LDC' in region:
                ldc_count += 1
              if 'USA' in region:
                usa_count += 1
              if 'India' in region:
                india_count += 1
              if 'China' in region:
                china_count += 1
                
        counts = [annex1_count, non_annex1_count, ldc_count, usa_count, india_count, china_count]
    
    else:
        counts = [
            getattr(data_grouped[data_grouped['ResearchRegion'].apply(lambda x: cat in x)][col], agg_func)()
            for cat in categories
        ]
    
    groupings = ['Region', 'Region', 'Region', 'Country', 'Country', 'Country']
    
    plot_data = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Grouping': groupings
    })
    
    # Plot using Plotly Express
    fig = px.bar(
        plot_data,
        x='Category',
        y='Count',
        color='Grouping',
        title=col_name+' Distribution for UNFCCC Groupings',
        labels={'Category': 'Group / Country', 'Count': col_name},
        color_discrete_map={'Region': 'steelblue', 'Country': 'lightcoral'}
    )
    
    # Customize layout
    fig.update_layout(
        legend_title_text='Grouping',
        xaxis_title='Group / Country',
        yaxis_title='Number of ' + col_name,
        yaxis=dict(gridcolor='#e3e3e3'),
    )
    
    st.plotly_chart(fig)
    
    
def plot_line(data_grouped, col, agg_func, col_name):
    # List of regions and countries
    research_regions = ['Annex1', 'NonAnnex1', 'LDC', 'USA', 'China', 'India']
    
    plot_data = []
    
    if col == 'Authors':
        
        data_grouped['AuthorCountry'] = data_grouped['AuthorCountry'].apply(ast.literal_eval)
        
        # Dictionaries to store yearly counts
        region_yearly_counts = {
            'Annex1': defaultdict(int),
            'NonAnnex1': defaultdict(int),
            'LDC': defaultdict(int),
            'USA': defaultdict(int),
            'China': defaultdict(int),
            'India': defaultdict(int),
        }
        
        # Iterate through the AuthorCountry column to update the counters
        for idx, row in data_grouped.iterrows():
            year = row['Year Published']
            author_dict = row['AuthorCountry']
            for author, country in author_dict.items():
                if country:  # Ensure the country is not None
                    country = clean_regions(country)[0]
                    region = get_research_region(country)
                    for key in region_yearly_counts.keys():
                        if key in region:
                            region_yearly_counts[key][year] += 1
        
        # Convert counts into DataFrames and append to plot_data
        years = sorted(set(data_grouped['Year Published']))
        for region, counts in region_yearly_counts.items():
            region_counts = pd.DataFrame({
                'Year Published': years,
                'Count': [counts[year] for year in years],
                'ResearchRegion': region
            })
            plot_data.append(region_counts)
        
    else:
    
        # Group data by year and research region
        for research_region in research_regions:
            counts = (data_grouped[data_grouped['ResearchRegion'].apply(lambda x: research_region in x)]
                          .groupby('Year Published')[col].agg(agg_func).reset_index(name='Count'))
            if col_name == 'Cumulative Citations':
                counts['Count'] = counts['Count'].cumsum()
            counts['ResearchRegion'] = research_region
            plot_data.append(counts)
        
    
    # Combine all regions into a single DataFrame
    plot_data = pd.concat(plot_data, ignore_index=True)
    
    # Determine line style based on region type
    plot_data['LineStyle'] = plot_data['ResearchRegion'].apply(
        lambda x: 'dash' if x in ['USA', 'China', 'India'] else 'solid'
    )
    plot_data['Marker'] = plot_data['ResearchRegion'].apply(
        lambda x: 'diamond' if x in ['USA', 'China', 'India'] else 'circle'
    )
    
    # Create an interactive line plot
    fig = px.line(
        plot_data,
        x='Year Published',
        y='Count',
        color='ResearchRegion',
        line_dash='LineStyle',
        symbol='Marker',
        title='Distribution of ' + col_name + ' over the Years for UNFCCC Groupings',
        labels={'Year Published': 'Year Published', 'Count': 'Total ' + col_name, 'ResearchRegion': 'Grouping'},
        hover_data={'LineStyle': False, 'Marker': False}
    )
    
    # Update layout for customization
    fig.update_layout(
        legend_title_text='Grouping',
        xaxis_title='Year Published',
        yaxis_title='Total ' + col_name,
        xaxis=dict(tickmode='linear'),  # Ensure all years appear
        yaxis=dict(gridcolor='#e3e3e3'),
        plot_bgcolor='white',
        legend=dict(
            title_font_size=11,
            font_size=11,
            tracegroupgap=10,  # Space between grouped legends
        ),
        hoverlabel=dict(
            bgcolor="white",  # Background color for hover box
            font_size=11,
        ),
    )
    
    # Simplify legend labels
    fig.for_each_trace(lambda trace: trace.update(name=trace.name.split(",")[0], showlegend=True,
                                                  mode='lines+markers'))
    
    st.plotly_chart(fig)
    


def plot_scatter(data_country_grouped, x_col, y_col):
    
    # Assuming temp has columns 'NumPublications', 'population', 'ResearchRegions', and 'Country'
    fig = px.scatter(
        data_frame=data_country_grouped,
        x=x_col,
        y=y_col,
        color='ResearchRegion',
        hover_name='Regions_updated',  # This will show the country name on hover
        category_orders={'ResearchRegion': ['Annex1', 'NonAnnex1', 'LDC', 'China', 'India', 'USA']},  # Ensures the order of colors in the legend
        size_max=None,  # Adjust the size of markers if needed
        title=y_col + " vs " + x_col
    )
    
    
    region_mapping = {
        'NonAnnex1LDC': 'LDC',
        'NonAnnex1China': 'China',
        'NonAnnex1India': 'India',
        'Annex1USA': 'USA'
    }
    
    # Manually update the legend labels
    for legend_entry in fig['data']:
        legend_entry['name'] = region_mapping.get(legend_entry['name'], legend_entry['name'])
        
    # Update the layout for white background and custom gridlines
    fig.update_layout(
        plot_bgcolor='white',  # Background inside the plot
        paper_bgcolor='white',  # Background outside the plot
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="UNFCCC Groupings / Countries",
        font=dict(color='black'),  # Ensure text is visible
        xaxis=dict(
            showgrid=True,  # Show gridlines for the x-axis
            gridcolor='lightgrey',  # Gridline color
            zerolinecolor='grey'  # Zero-line color (if applicable)
        ),
        yaxis=dict(
            showgrid=True,  # Show gridlines for the y-axis
            gridcolor='lightgrey',  # Gridline color
            zerolinecolor='grey'  # Zero-line color (if applicable)
        ),
        width=800,  # Set the width of the plot
        height=600
    )
    
    # Update marker size (control marker size explicitly)
    fig.update_traces(marker=dict(size=10))  # Increase the marker size
    
    # Show the plot
    #fig.show()
    st.plotly_chart(fig)
    
def plot_grouped_bar(data_grouped):
    
    # Prepare the data in long format
    data_journal = data_grouped[['Journal', 'ResearchRegion', 'DOI']]
    data_journal = data_journal.explode('ResearchRegion')
    data_journal = data_journal.groupby(['Journal', 'ResearchRegion'])['DOI'].count().reset_index()
    data_journal.columns = ['Journal', 'ResearchRegion', 'RegionCounts']
    
    data_journal_norm = data_journal.groupby('Journal')['RegionCounts'].sum().reset_index()
    data_journal_norm.columns = ['Journal', 'JournalCounts']
    
    data_journal = pd.merge(data_journal, data_journal_norm, on='Journal', how='left')
    data_journal['NormalizedCount'] = data_journal['RegionCounts'] / data_journal['JournalCounts']
    
    # Grouped bar chart
    fig = go.Figure()
    
    regions = ['Annex1', 'NonAnnex1', 'LDC', 'USA', 'China', 'India']
    hatch_patterns = {'USA': '\\', 'China': '\\', 'India': '\\', 'Annex1': None, 'NonAnnex1': None, 'LDC': None}
    
    # Create a bar for each region
    for region in regions:
        region_data = data_journal[data_journal['ResearchRegion'] == region]
        fig.add_trace(go.Bar(
            x=region_data['Journal'],
            y=region_data['NormalizedCount'],
            name=region,
            marker=dict(
                pattern_shape=hatch_patterns[region],  # Apply hatch pattern for specific regions
                line=dict(color='black', width=1)
            )
        ))
    
    # Layout for the plot
    fig.update_layout(
        barmode='group',
        title='Journal-Wise Distribution of Normalized Publications for UNFCCC Groupings',
        xaxis=dict(title='Journal', tickangle=90),
        yaxis=dict(title='Normalized Count of Publications'),
        legend_title_text='Grouping',
        template='plotly_white',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig)
    
def plot_heatmap(data_grouped):
    data_grouped['AuthorCountry'] = data_grouped['AuthorCountry'].apply(ast.literal_eval)
    # Calculate the intersection counts
    intersection_counter = Counter()
    for idx, row in data_grouped.iterrows():
        all_research_region = row['ResearchRegion']
        author_dict = row['AuthorCountry']
        for author, country in author_dict.items():
            if country:
                country = clean_regions(country)[0]
                all_author_region = get_research_region(country)
                for research_region in all_research_region:
                    for author_region in all_author_region:
                        intersection_counter[(research_region, author_region)] += 1
    
    # Create the matrix
    matrix = np.zeros((len(research_regions), len(research_regions)))
    for i, research in enumerate(research_regions):
        for j, author in enumerate(research_regions):
            matrix[i, j] = intersection_counter.get((research, author), 0)
    
    # Flip and round the matrix
    matrix_flipped = np.flipud(matrix)
    matrix_rounded = np.round(np.sqrt(matrix_flipped), 0)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_rounded,
        x=research_regions,
        y=research_regions[::-1],
        colorscale='YlOrRd',
        colorbar=dict(
            title="Number of Publications",
            tickvals=np.linspace(0, np.max(matrix_rounded), 6),  # Adjust the number of ticks
            ticktext=[str(int(i ** 2)) for i in np.linspace(0, np.max(matrix_rounded), 6)],  # Display original scale labels
            ticks="outside",  # Place ticks outside the color bar
            tickangle=0  # Ensure tick labels are horizontally aligned
        ),
        text=matrix_flipped,  # Values to display inside the boxes
        hoverinfo='x+y+text',  # Show the text (numbers) when hovering
        texttemplate='%{text}',
    ))
    
    # Set layout and axis labels
    fig.update_layout(
        title="Heatmap of Publications: Research Groups vs Author Groups",
        xaxis=dict(title="Author Group", ticks=""),
        yaxis=dict(title="Research Group", ticks=""),
        xaxis_nticks=len(research_regions),  # Set the number of ticks on the x-axis
        yaxis_nticks=len(research_regions),  # Set the number of ticks on the y-axis
        template="plotly_white",
        height=600,  # Set height and width to make it square
        width=1000,
        #autosize=False, 
        showlegend=False  # Hides the color legend
    )
    
    st.plotly_chart(fig)
    
def plot_world_map(data_grouped, vulnerability, population):
    data_pub = data_grouped[['Regions_updated', 'ResearchRegion', 'DOI', 'Altmetric']]
    data_pub = data_pub.explode('Regions_updated')
    data_pub = data_pub[~data_pub['Regions_updated'].isin(exclude_regions)]
    data_pub = data_pub.groupby('Regions_updated').agg({'DOI': 'count', 'Altmetric': 'sum'}).reset_index()
    data_pub.columns = ['Regions_updated', 'NumPublications', 'Altmetrics']
    
    # Load world map shapefile
    world = gpd.read_file("config/shapefiles/ne_50m_admin_0_countries.shp")
    world['Regions_updated'] = world['NAME'].apply(clean_regions)
    world = world.explode('Regions_updated')
    world = world.merge(vulnerability, on='Regions_updated', how='left')
    world.loc[world['NAME'] == 'S. Sudan', 'VulnerabilityIndex'] = list(world.loc[world['NAME'] == 'Sudan', 
                                                                                  'VulnerabilityIndex'])[0]
    
    world = world.merge(data_pub, on='Regions_updated', how='left')
    world = world.merge(population, on='Regions_updated', how='left')
    world['ResearchRegion'] = world['Regions_updated'].apply(get_singular_region)
    
    # Plot the world map with ND-GAIN Vulnerability Index
    fig, ax = plt.subplots(1, 1, figsize=(26, 18))
    
    # Create the map with a color scheme for the vulnerability index
    world.plot(column='VulnerabilityIndex', ax=ax, legend=True,
               cmap='OrRd', missing_kwds={'color': 'white'},
               legend_kwds={'label': "ND-GAIN Vulnerability Index",
                            'orientation': "vertical",
                            'shrink': 0.5, # Shrink the height of the legend
                            'pad': 0.03})
    
    hatch_patterns = {'Annex1': '///', 'NonAnnex1': '...', 'LDC': '++', None:''}
    
    # Apply hatching based on ResearchRegion
    for idx, row in world.iterrows():
        category = row['ResearchRegion']
        hatch = hatch_patterns.get(category, '')
    
        # Plot each country with the corresponding hatching pattern
        world[world.index == idx].plot(
            ax=ax,
            color='none',  # No fill color, just hatch
            hatch=hatch,
            facecolor='darkgrey',
            alpha=0.3
        )
    
    # Define marker shapes for each category
    markers = {
        'Annex1': 'D',
        'NonAnnex1': 's',
        'LDC': '^',
        'USA': 'P',
        'India': 'X',
        'China': '*',
        'None': 'o'
    }
    
    # Use a sequential colormap for the number of publications
    cmap = plt.colormaps.get_cmap('Blues')  # You can choose other sequential colormaps like 'Reds', 'Greens', etc.
    norm = Normalize(vmin=world['NumPublications'].min(), vmax=world['NumPublications'].max())  # Normalize publication counts
    
    # Add circles for publications, where size is based on population and color on normalized publication count
    for idx, row in world.iterrows():
        if pd.notna(row['NumPublications']):
    
            category = row['ResearchRegion']
            marker = markers.get(category, 'o')  # Default to circle if not found
    
            # Get the color based on the number of publications
            circle_color = cmap(norm(row['NumPublications']))
    
            ax.scatter(
                row['geometry'].centroid.x,
                row['geometry'].centroid.y,
                s=row['Population'] / 1000000,  # Scale population for circle size
                color=circle_color,  # Use color from the colormap
                edgecolor='black',
                alpha=0.7,
                # marker=marker
            )
    
    # Create colorbar to reflect the publication counts
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array(world['NumPublications']))  # Set array for ScalarMappable
    
    # Place the colorbar at the bottom, horizontally
    # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.4, pad=0.05)  # 'pad' adjusts the distance from the map
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)  # 'pad' adjusts the distance from the map
    cbar.set_label('Number of Publications')  # Add label to the colorbar
    
    # Create custom legend handles for each hatch pattern
    hatch_legend_handles = [
        mpatches.Patch(facecolor='white', hatch=hatch, edgecolor='black', label=category)
        for category, hatch in hatch_patterns.items() if hatch  # Exclude 'None' category or others without hatching
    ]
    
    # Add the hatch legend to the plot
    ax.legend(handles=hatch_legend_handles, title="UNFCCC Grouping", loc='lower right', frameon=True,
        labelspacing=0.5,
        borderpad=0.8,
        handleheight=2.5,
        handlelength=2.5)
    
    # Show the final plot
    plt.title("Publications over the world with Vulnerability Index")
    st.pyplot(fig)
    
def plot_interactive_world_map(data_grouped, vulnerability, population):
    
    data_pub = data_grouped[['Regions_updated', 'ResearchRegion', 'DOI', 'Altmetric']]
    data_pub = data_pub.explode('Regions_updated')
    data_pub = data_pub[~data_pub['Regions_updated'].isin(exclude_regions)]
    data_pub = data_pub.groupby('Regions_updated').agg({'DOI': 'count', 'Altmetric': 'sum'}).reset_index()
    data_pub.columns = ['Regions_updated', 'NumPublications', 'Altmetrics']
    
    # Load world map shapefile
    world = gpd.read_file("config/shapefiles/ne_50m_admin_0_countries.shp")
    world['Regions_updated'] = world['NAME'].apply(clean_regions)
    world = world.explode('Regions_updated')
    world = world.merge(vulnerability, on='Regions_updated', how='left')
    world.loc[world['NAME'] == 'S. Sudan', 'VulnerabilityIndex'] = list(world.loc[world['NAME'] == 'Sudan', 
                                                                                  'VulnerabilityIndex'])[0]
    
    world = world.merge(data_pub, on='Regions_updated', how='left')
    world = world.merge(population, on='Regions_updated', how='left')
    world['ResearchRegion'] = world['Regions_updated'].apply(get_singular_region)

    world = world[world['VulnerabilityIndex'].isna() == False]
    world['NumPublications'] = world['NumPublications'].fillna(0)
    world['Population'] = world['Population'].fillna(0)
    
    fig = go.Figure()
    
    # Create a list for the custom hover text
    world['hover_text'] = (
        "Country: " + world['Regions_updated'].str.title() +
        "<br>Population: " + world['Population'].apply(lambda x: '{:,}'.format(x)) +
        "<br>Number of Publications: " + world['NumPublications'].astype(str) +
        "<br>Vulnerability Index: " + world['VulnerabilityIndex'].apply(lambda x: '{:.3f}'.format(x)) +
        "<br>UNFCCC Grouping: " + world['ResearchRegion'].astype(str)
    )
    
    fig.add_trace(go.Choropleth(
        locations=world['Regions_updated'],  # Match country names to Plotly's built-in geo map
        z=world['VulnerabilityIndex'],
        locationmode='country names',
        colorscale='YlOrRd',
        colorbar=dict(
            title=dict(
                text="ND-GAIN Vulnerability Index",  # Title text
                side="right"  # Title position (default is 'right')
            ),
            ticks='outside',  # Style the ticks
            x=1,  # Adjust position horizontally (closer to the map)
            y=0.5,  # Center vertically
        ),
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title="Publications Over the World with Vulnerability Index",
        geo=dict(
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        margin={"r":0, "t":50, "l":0, "b":0},  # Reduce margins for better fit
    )
    
    fig.add_trace(go.Scattergeo(
        locations=world['Regions_updated'],
        locationmode='country names',
        mode='markers',
        text=world['hover_text'],  # Add custom hover text
        hoverinfo='text',
        marker=dict(
            size=world['Population']/30000000,  # Scale population for size
            color=world['NumPublications'],
            colorscale='Blues',
            cmin=world['NumPublications'].min(),  # Min publication value
            cmax=world['NumPublications'].max(),
            line=dict(color='black', width=0.5),
            colorbar=dict(
                title="Number of Publications",
                ticks='outside',
                x=0.5,  # Center it horizontally
                y=-0.1,  # Position below the map
                xanchor='center',  # Horizontal alignment of the colorbar
                yanchor='top',
                orientation='h',
                len=0.6))))


    st.plotly_chart(fig)

    
    

def main():
    data_path = "data/preprocessed_data.parquet"
    vulnerability_data_path = "data/vulnerability.csv"
    population_data_path = "data/population_projection.csv"
    
    data = load_data(data_path)
    vulnerability = read_vulnerability_data(vulnerability_data_path)
    population = read_population_data(population_data_path)
    
    data_country_grouped = prepare_graph_data(data, vulnerability, population)
    
    st.title("Equity in Climate Change Research")

    st.sidebar.title("Select Graphs")
    main_category = st.sidebar.selectbox("Choose a category:", 
                                         ["Scatter Plots", "Region Plots", "Line Plots", "Other Plots"])
    
    
    if main_category == "Scatter Plots":
    
        # Sidebar Navigation
        nav_option = st.sidebar.radio(
            "Select a graph to view:",
            ("NumPublications vs Vulnerability", "NumPublications vs Population", "Altmetrics vs Vulnerability")
        )
        
        # Display the selected graph
        if nav_option == "NumPublications vs Vulnerability":
            plot_scatter(data_country_grouped, "VulnerabilityIndex", "NumPublications_sqrt")
        elif nav_option == "NumPublications vs Population":
            plot_scatter(data_country_grouped, "Population", "NumPublications_sqrt")
        elif nav_option == "Altmetrics vs Vulnerability":
            plot_scatter(data_country_grouped, "VulnerabilityIndex", "Altmetrics_sqrt")
            
    elif main_category == "Region Plots":
    
        # Sidebar Navigation
        nav_option = st.sidebar.radio(
            "Select a graph to view:",
            ("NumPublications", "Citations", "Altmetrics", "Authors")
        )
        
        # Display the selected graph
        if nav_option == "NumPublications":
            plot_bar(data, 'DOI', 'count', "Publication")
        elif nav_option == "Citations":
            plot_bar(data, 'Citation', 'sum', "Citations")
        elif nav_option == "Altmetrics":
            plot_bar(data, 'Altmetric', 'sum', "Altmetrics")
        elif nav_option == "Authors":
            plot_bar(data, 'Author', 'sum', 'Author')
            
    elif main_category == 'Line Plots':
        
        # Sidebar Navigation
        nav_option = st.sidebar.radio(
            "Select a graph to view:",
            ("NumPublications", "Citations", "Cumulative Citations", "Altmetrics", "Authors")
        )
        
        # Display the selected graph
        if nav_option == "NumPublications":
            plot_line(data, 'DOI', 'count', 'Publications')
        elif nav_option == "Citations":
            plot_line(data, 'Citation', 'sum', 'Citations')
        elif nav_option == "Cumulative Citations":
            plot_line(data, 'Citation', 'sum', 'Cumulative Citations')
        elif nav_option == "Altmetrics":
            plot_line(data, 'Altmetric', 'sum', "Altmetrics")
        elif nav_option == "Authors":
            plot_line(data, 'Authors', 'sum', "Authors")
            
    elif main_category == "Other Plots":
        
        # Sidebar Navigation
        nav_option = st.sidebar.radio(
            "Select a graph to view:",
            ("Author Heatmaps", "Journal Distribution", "World Map")
        )
        
        if nav_option == "Journal Distribution":
            plot_grouped_bar(data)
        elif nav_option == "Author Heatmaps":
            plot_heatmap(data)
        elif nav_option == "World Map":
            plot_interactive_world_map(data, vulnerability, population)

    
    
    
if __name__ == "__main__":
    main()