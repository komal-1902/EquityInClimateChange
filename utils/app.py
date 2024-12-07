#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:33:51 2024

@author: komal
"""

import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config.regions as rgn
import helper_functions as func
import text_content as tc

@st.cache_data
def load_data(file_path):
    """Load preprocessed data."""
    return pd.read_parquet(file_path)

def prepare_graph_data(data_grouped, vulnerability, population):
    data_country = data_grouped[['Regions_updated', 'ResearchRegion', 'DOI', 'Altmetric']]
    data_country = data_country.explode('Regions_updated')
    
    data_country = data_country[~data_country['Regions_updated'].isin(rgn.exclude_regions)]
    
    data_country = pd.merge(data_country, vulnerability, on='Regions_updated', how='left')
    data_country = pd.merge(data_country, population, on='Regions_updated', how='left')
    data_country['VulnerabilityIndex'] = data_country['VulnerabilityIndex'].fillna(0)
    
    data_country['ResearchRegion'] = data_country['ResearchRegion'].apply(lambda x: repr(x.tolist()))
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
              country = func.clean_regions(country)[0]
              region = func.get_research_region(country)
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
    
    total_annex1 = counts[0]
    total_non_annex1 = counts[1]
    total = total_annex1 + total_non_annex1

    # Calculate percentages for hover data
    percentages = [count / total * 100 if total > 0 else 0 for count in counts]
    percentage_labels = [f"{percentage:.2f}%" for percentage in percentages]
    
    plot_data = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Grouping': groupings,
        'Percent of Total': percentage_labels
    })
    
    # Plot using Plotly Express
    fig = px.bar(
        plot_data,
        x='Category',
        y='Count',
        color='Grouping',
        title=col_name+' Distribution for UNFCCC Groupings',
        labels={'Category': 'Group / Country', 'Count': col_name},
        color_discrete_map={'Region': 'steelblue', 'Country': 'lightcoral'},
        hover_data={
            'Grouping': False,
            'Count': True,
            'Percent of Total': True
        }
    )
    
    # Customize layout
    fig.update_layout(
        legend_title_text='UNFCCC Grouping',
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
                    country = func.clean_regions(country)[0]
                    region = func.get_research_region(country)
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
    


def plot_scatter(data_country_grouped, x_col, y_col, col_name):
    
    data_country_grouped = data_country_grouped[data_country_grouped['ResearchRegion'].isin(['Annex1', 'NonAnnex1', 'LDC'])]
    data_country_grouped['Regions_updated'] = data_country_grouped['Regions_updated'].str.title()
    data_country_grouped = data_country_grouped.rename(columns={'ResearchRegion': 'UNFCCC Grouping'})
    
    # Assuming temp has columns 'NumPublications', 'population', 'ResearchRegions', and 'Country'
    fig = px.scatter(
        data_frame=data_country_grouped,
        x=x_col,
        y=y_col,
        color='UNFCCC Grouping',
        hover_name='Regions_updated',  # This will show the country name on hover
        hover_data={col_name: True},
        category_orders={'UNFCCC Grouping': ['Annex1', 'NonAnnex1', 'LDC']},  # Ensures the order of colors in the legend
        size_max=None,  # Adjust the size of markers if needed
        title=col_name + " vs " + x_col
    )
        
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
    st.plotly_chart(fig)
    
def plot_normalised_bar_plot(data_country_grouped):
    data_region_grouped = data_country_grouped.groupby('ResearchRegion').agg({'NumPublications': 'sum', 'Population': 'sum'}).reset_index()
    data_region_grouped['Normalised_Publications'] = data_region_grouped['NumPublications'] / data_region_grouped['Population']
    data_region_grouped = data_region_grouped.sort_values('Normalised_Publications', ascending=False)
    
    #print(data_region_grouped.head(10))
    
    region_order = ['Annex1', 'NonAnnex1', 'LDC', 'USA', 'China', 'India']
    #for region in region_order:
    #    print(data_region_grouped.loc[data_region_grouped['ResearchRegion'] == region, 'Normalised_Publications'])
    region_vals = [data_region_grouped.loc[data_region_grouped['ResearchRegion'] == region, 'Normalised_Publications'].values[0]
                    for region in region_order]
    region_vals_scaled = [val * 1e6 for val in region_vals]
    
    
    
    groupings = ['Region', 'Region', 'Region', 'Country', 'Country', 'Country']

    # Create a DataFrame
    plot_data = pd.DataFrame({
        'Category': region_order,
        'Count': region_vals_scaled,
        'Grouping': groupings
    })
    
    plot_data['Normalised Publications'] = plot_data['Count'].apply(lambda x: f"{x:.2f} x 10⁻⁶")
    
    # Plot using Plotly Express
    fig = px.bar(
        plot_data,
        x='Category',
        y='Count',
        color='Grouping',
        title='Population-Normalised Publication Distribution for UNFCCC Groupings',
        labels={'Category': 'UNFCCC Grouping', 'Count': 'Normalised Publications'},
        color_discrete_map={'Region': 'steelblue', 'Country': 'lightcoral'},
        hover_data = {
            'Grouping': False,
            'Normalised Publications': True,
        }
    )
    
    # Customize layout
    fig.update_layout(
        legend_title_text='Grouping',
        xaxis_title='Group / Country',
        yaxis_title='Normalised Number of Publications',
        yaxis=dict(gridcolor='#e3e3e3', tickformat='.1f'),
        plot_bgcolor='white',
        annotations=[
            # Add a text annotation to indicate the scale is in 10⁻⁶
            dict(
                x=0,  # Positioning the annotation in the center
                y=1.05,  # Slightly above the plot
                xref="paper",
                yref="paper",
                text="Scale: 10⁻⁶",  # Scale information
                showarrow=False,
                font=dict(size=12, color='grey'),
            )
        ]
    )
    
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
        
        # Custom hovertemplate
        hovertemplate = (
            "Journal: %{x}<br>"  # Display Journal on hover
            "UNFCCC Grouping: " + region + "<br>"  # Display the region from customdata
            "Normalized Publication Count: %{y:.3f}<br>"  # Display the normalized count
        )
        
        fig.add_trace(go.Bar(
            x=region_data['Journal'],
            y=region_data['NormalizedCount'],
            name=region,
            hovertemplate=hovertemplate,
            hoverinfo='skip',
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
        legend_title_text='UNFCCC Grouping',
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
                country = func.clean_regions(country)[0]
                all_author_region = func.get_research_region(country)
                for research_region in all_research_region:
                    for author_region in all_author_region:
                        intersection_counter[(research_region, author_region)] += 1
    
    # Create the matrix
    matrix = np.zeros((len(rgn.research_regions), len(rgn.research_regions)))
    for i, research in enumerate(rgn.research_regions):
        for j, author in enumerate(rgn.research_regions):
            matrix[i, j] = intersection_counter.get((research, author), 0)
    
    # Flip and round the matrix
    #matrix_flipped = np.flipud(matrix)
    matrix_rounded = np.round(np.sqrt(matrix), 0)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_rounded,
        x=rgn.research_regions,
        y=rgn.research_regions,
        colorscale='YlOrRd',
        colorbar=dict(
            title="Number of Publications",
            tickvals=np.linspace(0, np.max(matrix_rounded), 6),  # Adjust the number of ticks
            ticktext=[str(int(i ** 2)) for i in np.linspace(0, np.max(matrix_rounded), 6)],  # Display original scale labels
            ticks="outside",  # Place ticks outside the color bar
            tickangle=0  # Ensure tick labels are horizontally aligned
        ),
        text=matrix,  # Values to display inside the boxes
        hoverinfo='x+y+text',  # Show the text (numbers) when hovering
        hovertemplate=(
            'Author Group = %{x}<br>' +
            'Research Group = %{y}<br>' +
            'Correlation Count = %{text}<extra></extra>'
        ),  # Custom hover text format
        texttemplate='%{text}',
    ))
    
    # Set layout and axis labels
    fig.update_layout(
        title="Heatmap of Publications: Research Groups vs Author Groups",
        xaxis=dict(title="Author Group", ticks=""),
        yaxis=dict(title="Research Group", ticks=""),
        xaxis_nticks=len(rgn.research_regions),  # Set the number of ticks on the x-axis
        yaxis_nticks=len(rgn.research_regions),  # Set the number of ticks on the y-axis
        template="plotly_white",
        height=600,  # Set height and width to make it square
        width=1000,
        #autosize=False, 
        showlegend=False  # Hides the color legend
    )
    
    st.plotly_chart(fig)
    
def plot_interactive_world_map(data_grouped, vulnerability, population):
    
    data_pub = data_grouped[['Regions_updated', 'ResearchRegion', 'DOI', 'Altmetric']]
    data_pub = data_pub.explode('Regions_updated')
    data_pub = data_pub[~data_pub['Regions_updated'].isin(rgn.exclude_regions)]
    data_pub = data_pub.groupby('Regions_updated').agg({'DOI': 'count', 'Altmetric': 'sum'}).reset_index()
    data_pub.columns = ['Regions_updated', 'NumPublications', 'Altmetrics']
    
    # Load world map shapefile
    world = gpd.read_file("config/shapefiles/ne_50m_admin_0_countries.shp")
    world['Regions_updated'] = world['NAME'].apply(func.clean_regions)
    world = world.explode('Regions_updated')
    world = world.merge(vulnerability, on='Regions_updated', how='left')
    world.loc[world['NAME'] == 'S. Sudan', 'VulnerabilityIndex'] = list(world.loc[world['NAME'] == 'Sudan', 
                                                                                  'VulnerabilityIndex'])[0]
    
    world = world.merge(data_pub, on='Regions_updated', how='left')
    world = world.merge(population, on='Regions_updated', how='left')
    world['ResearchRegion'] = world['Regions_updated'].apply(func.get_singular_region)

    world['NumPublications'] = world['NumPublications'].fillna(0)
    world['Population'] = world['Population'].fillna(0)
    
    fig = go.Figure()
    
    world_vul = world[world['VulnerabilityIndex'].isna() == False]
    world['VulnerabilityIndex'] = world['VulnerabilityIndex'].fillna(0)
    
    # Create a list for the custom hover text
    world['hover_text'] = (
        "Country: " + world['Regions_updated'].str.title() +
        "<br>Population: " + world['Population'].apply(lambda x: '{:,}'.format(x) if x != 0 else 'NA') +
        "<br>Number of Publications: " + world['NumPublications'].astype(str) +
        "<br>Vulnerability Index: " + world['VulnerabilityIndex'].apply(lambda x: '{:.3f}'.format(x) if x != 0 else 'NA') +
        "<br>UNFCCC Grouping: " + world['ResearchRegion'].apply(lambda x: x if x != '' else 'NA')
    )
    
    fig.add_trace(go.Choropleth(
        locations=world_vul['Regions_updated'],  # Match country names to Plotly's built-in geo map
        z=world_vul['VulnerabilityIndex'],
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
        hoverinfo='none',
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
                len=0.8)
            )
        )
        )

    st.plotly_chart(fig)
    

def main():
    data_path = "data/preprocessed_data.parquet"
    vulnerability_data_path = "data/vulnerability.csv"
    population_data_path = "data/population_projection.csv"
    
    data = load_data(data_path)
    vulnerability = func.read_vulnerability_data(vulnerability_data_path)
    population = func.read_population_data(population_data_path)
    
    data_country_grouped = prepare_graph_data(data, vulnerability, population)
    
    st.title("Equity in Climate Change Research")
    

    st.sidebar.title("Graphs")
    st.sidebar.info("Use this menu to select the type of graph you'd like to view.")
    
    main_category = st.sidebar.selectbox("Choose Graph:", 
                                         ["Publication Analysis", "Citation Analysis", "Altmetric Analysis",
                                          "Author Analysis"])
    
    if main_category == "Publication Analysis":
        nav_option = st.sidebar.radio("Select a graph to view:",
                                        ("Publications Distributed over the World", 
                                         "Publications of Countries with their Vulnerability", 
                                         "Publications of Countries with their Population",
                                         "Publication Distributed over Groups", 
                                         "Population-Normalized Publication Distribution over Groups",
                                         "Publication Distributed over Time",
                                         "Publications Distributed over Research Journals"))
        
        if nav_option == "Publications Distributed over the World":
            plot_interactive_world_map(data, vulnerability, population)
            
            st.subheader("How the plot works")
            st.write(tc.world_map_working)
            
            st.header("Key Takeaways")
            st.write(tc.world_map_takeaway)
                     
        elif nav_option == "Publications of Countries with their Vulnerability":
            plot_scatter(data_country_grouped, "VulnerabilityIndex", "NumPublications_sqrt", "NumPublications")
            
            st.subheader("How the plot works")
            st.write(tc.publication_vs_vulnerability_working)
            
            st.header("Key Takeaways")
            st.write(tc.publication_vs_vulnerability_takeaway)
            
        elif nav_option == "Publications of Countries with their Population":
            plot_scatter(data_country_grouped, "Population", "NumPublications_sqrt", "NumPublications")
            
            st.subheader("How the plot works")
            st.write(tc.publication_vs_population_working)
            
            st.header("Key Takeaways")
            st.write(tc.publication_vs_population_takeaway)
            
        elif nav_option == "Publication Distributed over Groups":
            plot_bar(data, 'DOI', 'count', "Publication")
            
            st.subheader("How the plot works")
            st.write(tc.region_wise_publication_working)
            
            st.header("Key Takeaways")
            st.write(tc.region_wise_publication_takeaway)
            
        elif nav_option == "Population-Normalized Publication Distribution over Groups":
            plot_normalised_bar_plot(data_country_grouped)
            
            st.subheader("How the plot works")
            st.write(tc.norm_region_wise_publication_working)
            
            st.header("Key Takeaways")
            st.write(tc.norm_region_wise_publication_takeaway)
            
        elif nav_option == "Publication Distributed over Time":
            plot_line(data, 'DOI', 'count', 'Publications')
            
            st.subheader("How the plot works")
            st.write(tc.year_wise_publication_working)
            
            st.header("Key Takeaways")
            st.write(tc.year_wise_publication_takeaway)
            
        elif nav_option == "Publications Distributed over Research Journals":
            plot_grouped_bar(data)

            st.subheader("How the plot works")
            st.write(tc.journal_wise_publication_working)
            
            st.header("Key Takeaways")
            st.write(tc.journal_wise_publication_takeaway)
            
    elif main_category == "Citation Analysis":
        nav_option = st.sidebar.radio("Select a graph to view:",
                                        ("Citations Distributed over Groups", "Citations Distributed over Time",
                                         "Cumulative Citations Distributed over Time"))
        
        if nav_option == "Citations Distributed over Groups":
            plot_bar(data, 'Citation', 'sum', "Citations")
            
            st.subheader("How the plot works")
            st.write(tc.region_wise_citation_working)
            
            st.header("Key Takeaways")
            st.write(tc.region_wise_citation_takeaway)
            
        elif nav_option == "Citations Distributed over Time":
            plot_line(data, 'Citation', 'sum', 'Citations')
            
            st.subheader("How the plot works")
            st.write(tc.year_wise_citation_working)
            
            st.header("Key Takeaways")
            st.write(tc.year_wise_citation_takeaway)
            
        elif nav_option == "Cumulative Citations Distributed over Time":
            plot_line(data, 'Citation', 'sum', 'Cumulative Citations')
            
            st.subheader("How the plot works")
            st.write(tc.cumulative_year_wise_citation_working)
            
            st.header("Key Takeaways")
            st.write(tc.cumulative_year_wise_citation_takeaway)
            
    elif main_category == "Altmetric Analysis":
        nav_option = st.sidebar.radio("Select a graph to view:",
                                        ("Altmetrics Distributed over Groups", 
                                         "Altmetrics Distributed over Time",
                                         "Altmetrics of Countries with their Vulnerability"))
        
        if nav_option == "Altmetrics Distributed over Groups":
            plot_bar(data, 'Altmetric', 'sum', "Altmetrics")
            
            st.subheader("How the plot works")
            st.write(tc.region_wise_altmetric_working)
            
            st.header("Key Takeaways")
            st.write(tc.region_wise_altmetric_takeaway)
            
        elif nav_option == "Altmetrics Distributed over Time":
            plot_line(data, 'Altmetric', 'sum', "Altmetrics")
            
            st.subheader("How the plot works")
            st.write(tc.year_wise_altmetric_working)
            
            st.header("Key Takeaways")
            st.write(tc.year_wise_altmetric_takeaway)
            
        elif nav_option == "Altmetrics of Countries with their Vulnerability":
            plot_scatter(data_country_grouped, "VulnerabilityIndex", "Altmetrics_sqrt", "Altmetrics")
            
            st.subheader("How the plot works")
            st.write(tc.altmetric_vs_vulnerability_working)
            
            st.header("Key Takeaways")
            st.write(tc.altmetric_vs_vulnerability_takeaway)
            
    elif main_category == "Author Analysis":
        nav_option = st.sidebar.radio("Select a graph to view:",
                                        ("Number of Authors Distributed over Groups", 
                                         "Number of Authors Distributed over Time",
                                         "Author Correlation with Groups"))
        
        if nav_option == "Number of Authors Distributed over Groups":
            plot_bar(data, 'Author', 'sum', 'Author')
            
            st.subheader("How the plot works")
            st.write(tc.region_wise_author_working)
            
            st.header("Key Takeaways")
            st.write(tc.region_wise_author_takeaway)
            
        elif nav_option == "Number of Authors Distributed over Time":
           plot_line(data, 'Authors', 'sum', "Authors")
           
           st.subheader("How the plot works")
           st.write(tc.year_wise_author_working)
           
           st.header("Key Takeaways")
           st.write(tc.year_wise_author_takeaway)
            
        elif nav_option == "Author Correlation with Groups":
            plot_heatmap(data)
            
            st.subheader("How the plot works")
            st.write(tc.heatmap_author_working)
            
            st.header("Key Takeaways")
            st.write(tc.heatmap_author_takeaway)
            
    
    
if __name__ == "__main__":
    main()