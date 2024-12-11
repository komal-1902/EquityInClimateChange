#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:42:16 2024

@author: komal
"""
# ------------------------------------------------------------------------------------------------------
# World Map
# ------------------------------------------------------------------------------------------------------

world_map_working = """World Map shows countries with their ND-GAIN Vulnerability Index using a red colorbar, 
        along Publications of that country with circles. The size of the circle
        indicates the relative population strength of the country, and the colour depth indicates
        the number of publications using a blue colorscale.
        Countries that were not found in our dataset do not contain an associated circle. 
        \n**Note:** Taiwan and Hong Kong have no unique vulnerability index, and so only their 
        population values and number of publications are shown. 
        \n**Interact with the map:** Users can zoom over the map and use the hover functionatlity 
        to get all information about any particular country."""
                       
world_map_takeaway = """The most climate-vulnerable countries are the focus of the fewest publications in the top-tier,
         high-impact journals analyzed in our dataset. Countries across Africa are particularly 
         under-represented in high-impact climate change research given that many are among 
         the most vulnerable and/or fall under the UN LDC grouping. In fact, 
         many African countries have zero publications (shown as a lack of blue circles). 
         \nIn contrast, Annex-1 countries that display the least vulnerability 
         are the focus of a relatively high number of publications.
         This is especially true of the USA which contains more publications in 
         these top-tier, high-impact journals than any other country in the dataset (denoted
         by the dark blue colour). China and India had the next highest publication totals, 
         and both display higher vulnerability than the USA."""
         
# ------------------------------------------------------------------------------------------------------
# Scatter plot for Num Publications Vs Vulnerability Index
# ------------------------------------------------------------------------------------------------------
         
publication_vs_vulnerability_working = """The scatter plot shows countries as bubbles, represented by their 
         ND-GAIN Vulnerability Index on x-axis and number of publications on the y-axis. Color differentiates 
         Annex-1, nonAnnex-1 and LDC regions. USA (Annex-1), China (nonAnnex-1) and India (nonAnnex-1) are 
         highlighted in separate colors. The square root of Number of publications is used
         on the y-axis to account for better readability.
         \n**Note:** Taiwan and Hong Kong have no unique vulnerability index, and so only their 
         bubble lies with VulnerabilityIndex=0 along with their number of publications. 
         \n**Interact with the plot:** Users can pan into the scatter plots with the crop feature 
         and use the hover functionatlity over any bubble to get all information
         about the particular country. To zoom out, double-click on the plot."""
         
publication_vs_vulnerability_takeaway = """USA has the highest number of publications 
          while having a lower Vulnerability Index. While China and India also have a relatively high 
          number of publications, their Vulnerability Index is much higher compared to the USA. 
          USA-focused publications still dominate the dataset even when normalized by population. 
          \nKeeping these anomalies aside, we see most of the Annex-1 countries scattered amongst 
          lower Vulnerability Index and many of them have a higher number of publication in contrast
          to non-Annex1 countries, which have lower publication count and are more vulnerable. 
          Annex-1 countries are generally the focus of more publications even despite their lower 
          vulnerability, while LDC countries are to the extreme end of the vulnerability scale with 
          the lowest publication counts, which is a source of concern."""
          
# ------------------------------------------------------------------------------------------------------
# Scatter Plot for Num Publications vs Population
# ------------------------------------------------------------------------------------------------------
          
publication_vs_population_working = """The scatter plot shows countries as bubbles, represented by their 
         population on x-axis and number of publications on the y-axis. Color differentiates Annex-1, 
         nonAnnex-1 and LDC regions. USA (Annex-1), China (nonAnnex-1) and India (nonAnnex-1) are 
         highlighted in separate colors. The square root of Number of publications is used
         on the y-axis to account for better readability.
         \n**Interact with the plot:** Users can pan into the scatter plots with the crop feature 
         and use the hover functionatlity over any bubble to get all information
         about the particular country. To zoom out, double-click on the plot."""
         
         
publication_vs_population_takeaway = """Even with low population compared to many countries, the USA has the 
          highest number of publications in top-tier journals, showing the influence the USA has in
          climate change research. At the same time, The more populous China 
          and India also display a relatively high number of publications (though not as high as 
          the USA). India, with a population >1.4 billion, has a number of publications similar 
          to many less-populous Annex-1 countries such as Denmark (~6M) and Australia (~27M).
          \nAll countries are clustered amongst lower publication numbers. Within these, 
          the countries with higher publication numbers are Annex-1 countries and in contrast 
          countries with the lowest publication count and LDC countries, which shows discrepancy 
          in publication patterns within these groups."""
          
# ------------------------------------------------------------------------------------------------------
# Region-wise Overall Publications
# ------------------------------------------------------------------------------------------------------
          
region_wise_publication_working = """The bar plot shows overall publication counts for each UNFCCC Grouping,
          along with special case studies for USA, China and India.
          \n**Interact with the plot:** Users can use the hover functionatlity over any bar to 
          get all information about the particular country or group."""
          
region_wise_publication_takeaway = """At first glance, we can see that the number of publications are comparable between 
         Annex-1 and nonAnnex-1 countries. In aggregate, Annex-1 countries were the focus of the 
         similar number of studies in top tier, high-impact journals (across our dataset) as 
         nonAnnex-1 countries, while the UN LDC grouping was the focus of fewer than 20% of the 
         studies. 
         \n It's interesting to note that the USA alone was the focus of more studies than the entire
         LDC country group (18 countries). China and India were also the focus of a high number of 
         studies individually."""
         
# ------------------------------------------------------------------------------------------------------
# Normalised Region-wise Overall Publications
# ------------------------------------------------------------------------------------------------------
          
norm_region_wise_publication_working = """The bar plot shows publication counts for each UNFCCC Grouping 
          normalised by the population. To normalise, the total aggregation of publication counts of each 
          UNFCCC Grouping is divided by the total population of all countries found in our database within 
          each UNFCCC Grouping.
          \n**Interact with the plot:** Users can use the hover functionatlity over any bar to 
          get all information about the particular country or group."""
          
norm_region_wise_publication_takeaway = """When normalized by population, Annex-1 countries (and the USA in 
          particular) markedly outweigh publications from nonAnnex-1 and LDC countries, although we note 
          that this result is only just significant (p-value = 0.049) owing to the relatively small number 
          of countries in these samples. When normalized, the grouping of nonAnnex-1 countries and LDC are 
          shown to be the focus region in very few studies in the dataset. China, when considered alone for 
          its population, has a higher normalized number of publications than the nonAnnex-1 countries 
          combined."""
         
# ------------------------------------------------------------------------------------------------------
# Year-Wise for Overall Publications
# ------------------------------------------------------------------------------------------------------
         
year_wise_publication_working = """The line chart shows number of publications per group over our entire timeframe
          (2011-2023). USA, China and India are represented by dash lines as special case studies
          of these groups. 
          \n**Interact with the plot:** Users can use the hover functionality over any marker to 
          get all information about the group at that given year."""
          
year_wise_publication_takeaway = """The number of publications across most country groupings and countries has risen in 
          top-tier journals from 2011 through the end of 2023, although this has been marked by 
          periods of annual variability. The aggregate Annex-1 and nonAnnex-1 groupings have both 
          been the focus of significantly more studies since ~2020, with a greater number of 
          top-tier journal publications focused on nonAnnex-1 countries during this brief, 
          recent period. Studies focused on China and the USA both show notable increases between 
          2022 and 2023 in particular, outpacing the LDC countries and India."""
          
# ------------------------------------------------------------------------------------------------------
# Journal Wise Grouped Bar Plot
# ------------------------------------------------------------------------------------------------------

journal_wise_publication_working = """The grouped bar plot shows top-tier research journals of our study on 
         the x-axis. Each journal contains bars for our UNFCCC groupings, along with special case studies of USA,
         China and India. Users can use this graph to note patterns of how each journal publishes
         research on a specific UNFCCC grouping / country by comparing similar coloured bars across
         the journals. Users can also note the pattern of publications across UNFCCC Groupings 
         within each journal to study their composition. 
         \nTo normalise, we divided the total number of publications per journal and UNFCCC group by 
         the total number of publications in that journal. 
         \n**Interact with the plot:** Users can use the hover functionatlity over any bar to 
         get all information about the particular country, group or journal. Users can also
         zoom into the graph by using the crop functionality. To zoom out, double-click on the plot."""
         
journal_wise_publication_takeaway = """Most of the top-tier journals publish studies whose geographic foci are 
         distributed similarly  cross the various countries and country groupings. Nature and 
         Nature Climate Change tend to publish more studies focused on Annex-1 regions as a percentage 
         of their total climate change publications (included in our dataset) relative to all other 
         groupings/countries shown, while the other journals appear to have a great fraction of 
         nonAnnex-1 countries. Nature Climate Change, Nature Communications, and PNAS also show 
         substantially more publications focused on the USA relative to LDCs, India and China 
         and also relative to the other journals. All journals show similar numbers of studies 
         focused on the LDC grouping, while Nature Communications and NPJ show more studies focused 
         on China. """
         
# ------------------------------------------------------------------------------------------------------
# Region-wise Overall Citations
# ------------------------------------------------------------------------------------------------------

region_wise_citation_working = """The bar plot shows total number of citations for each UNFCCC Grouping,
          along with special case studies for USA, China and India.
          \n**Interact with the plot:** Users can use the hover functionality over any bar to 
          get all information about the particular country or group."""
         
region_wise_citation_takeaway = """Citations show a research publication's outreach to scientific communities
         and can be a useful trend to study for estimating impact of climate change research. 
         The total number of citations are essentially the sum total of citations across all research articles
         published per group in our time-frame, and these numbers seem comparable between Annex-1 and
         nonAnnex-1 countries, but much lower for the more vulnerable LDC countries, implying that the 
         research articles published on these vulnerable regions don't get cited enough when compared to 
         research articles published on annex-1 or nonAnnex-1 groups. The citation totals for USA alone exceed
         more than the LDC group countries. """
         
# ------------------------------------------------------------------------------------------------------
# Year-wise Overall Citations
# ------------------------------------------------------------------------------------------------------

year_wise_citation_working = """The line chart shows the total number of citations per group over our entire 
          timeframe (2011-2023). USA, China and India are represented by dash lines as special case studies
          of these groups. For a given marker, the total number of citations refers to the sum total citations
          of all the research articles published on countries belonging to the UNFCCC group for that year.
          \n**Interact with the plot:** Users can use the hover functionality over any marker to 
          get all information about the group at that given year. The crop functionality can also be used to
          zoom into any portion of the graph. To zoom-out, double-click anywhere on the graph."""
          
year_wise_citation_takeaway = """Citations show a research publication's outreach to scientific communities
         and can be a useful trend to study for estimating impact of climate change research. 
         From the plot, we see that the total number of citations across most country groupings and 
         countries are erratic across our time-frame, with significant peaks visible for both Annex-1 
         and nonAnnex-1 groups in 2016. The LDC group trend line mimics nonAnnex-1 for many years, 
         which is expected seeing as all LDC countries are inclusive in the nonAnnex-1 group. 
         2020 sees a peak in LDC and nonAnnex-1 groups, and its reassuring to see the citations for 
         non-Annex1 steadily track with the Annex-1 citations in the recent years. 
         Drops in the number of citations in the more recent years of 2020-2023 can be theorized to the lack
         of time for the research articles to gain enough coverage, given that citations may take
         months or years to accumulate. To account for this, users can take 
         a look at the Cumulative Citation distribution over time for a more comparable study. """
          
# ------------------------------------------------------------------------------------------------------
# Year-wise Cumulative Citations
# ------------------------------------------------------------------------------------------------------

cumulative_year_wise_citation_working = """The line chart shows the cumulative number of citations per group 
          over our entire timeframe (2011-2023). USA, China and India are represented by dash lines as 
          special case studies of these groups. For a given marker, the cumulative number of citations refers 
          to the sum total citations of all the research articles published on countries belonging to the UNFCCC 
          group. This sum is calculated cumulatively, i.e. as a running total from the initial year 
          (i.e. 2011) to the current year.
          \n**Interact with the plot:** Users can use the hover functionality over any marker to 
          get all information about the group at that given year. The crop functionality can also be used to
          zoom into any portion of the graph. To zoom-out, double-click anywhere on the graph."""
          
cumulative_year_wise_citation_takeaway = """The cumulative number of citations provide a comparable way to 
          study trend lines for the UNFCCC groupings and special case countries. The sum total of all citations 
          for the entire time frame is encapsulated with each passing year. Hence, 2023 shows the total citations
          accumulated by each UNFCCC grouping over the due course of our study's time-frame. 
          Overall, cumulative citation numbers are rising across countries and groupings, with the Annex-1 and 
          nonAnnex-1 in aggregate showing comparable levels of citations later in the record. 
          Studies focused on the USA show significantly higher cumulative citations than those focused on 
          the LDCs and China, and relative to these, India’s cumulative citations are lower.
          So while it is agreeable to see nonAnnex-1 groups gain commensurate reach as Annex-1 in recent years,
          LDC groups are yet to gain traction. This gap can be ascertained to the difference in publication
          numbers between these groups, since lower number of publications would entail lower overall citation
          numbers. """
         
# ------------------------------------------------------------------------------------------------------
# Region-wise Overall Altmetrics
# ------------------------------------------------------------------------------------------------------

region_wise_altmetric_working = """The bar plot shows total sum of altmetric scores of research articles
          for each UNFCCC Grouping, along with special case studies for USA, China and India.
          \n**Interact with the plot:** Users can use the hover functionality over any bar to 
          get all information about the particular country or group."""
          
region_wise_altmetric_takeaway = """The total Altmetric Scores reach a soaring number for Annex-1 countries
          and nonAnnex-1 countries are not as close behind, although we see comparable difference in the numbers
          when we compare to overall number of publications or citations (users can navigate to see the mentioned
          graphs using left navigation bar). LDC groups have an overwhelmingly low total of Altmetric Scores, 
          much of which could be attributed to the overall number of publications being pretty low for LDC
          groups compared to the other two. USA exceeds the LDC groups in Altmetric totals, and users can
          appreciate that it reaches almost 50% of the score totals of the entire nonAnnex-1 group. 
          China and India have lower Altmetric Score totals compared to the rest of the groups. """
          
# ------------------------------------------------------------------------------------------------------
# Year-wise Overall Altmetrics
# ------------------------------------------------------------------------------------------------------
         
year_wise_altmetric_working = """The line chart shows the total sum of Altmetric Scores per group over our entire 
          timeframe (2011-2023). USA, China and India are represented by dash lines as special case studies
          of these groups. For a given marker, the 'Total Altmetrics' refers to the sum total Altmetric Scores
          of all the research articles published on countries belonging to a UNFCCC group for the current year.
          \n**Interact with the plot:** Users can use the hover functionality over any marker to 
          get all information about the group at that given year. The crop functionality can also be used to
          zoom into any portion of the graph. To zoom-out, double-click anywhere on the graph."""

year_wise_altmetric_takeaway = """ Altmetric scores measure outreach of a research publication using 
         non-traditional sources like social media, news, blogs and so on. They form a measure of the
         instantaneous attention that publications receive and therefore form an interesting comparative metric.
         From our plot, the instantaneous representation of press and public interest is shown in the annual 
         values of Altmetrics scores aggregated by grouping or country, most of which appear to be 
         experiencing growth over the last decade. For most years, aggregate Altmetrics scores are higher 
         for studies of Annex-1 focused compared to other groupings, although in recent years nonAnnex-1 
         studies are commanding more attention. The USA shows consistently higher Altmetrics aggregate 
         scores relative to China and India, albeit with relatively high interannual variability since ~2017. 
         Over this period, LDC countries have shown a substantial increase in Altmetric scores, while China 
         and India specifically have stayed relatively constant (albeit with China-focused studies showing 
         decreases since 2019 and then an increase since 2022). """
         
# ------------------------------------------------------------------------------------------------------
# Scatter plot for Altmetrics Total Vs Vulnerability Index
# ------------------------------------------------------------------------------------------------------
         
altmetric_vs_vulnerability_working = """The scatter plot shows countries as bubbles, represented by their 
         ND-GAIN Vulnerability Index on x-axis and total Altmetric Scores on the y-axis. Color differentiates 
         Annex-1, nonAnnex-1 and LDC regions. USA (Annex-1), China (nonAnnex-1) and India (nonAnnex-1) are 
         highlighted in separate colors. The square root of Altmetric Scores is used
         on the y-axis to account for better readability.
         \n**Note:** Taiwan and Hong Kong have no unique vulnerability index, and so only their 
         bubble lies with VulnerabilityIndex=0 along with their Altmetric Totals. 
         \n**Interact with the plot:** Users can pan into the scatter plots with the crop feature 
         and use the hover functionatlity over any bubble to get all information
         about the particular country. To zoom out, double-click on the plot."""
         
altmetric_vs_vulnerability_takeaway = """Publications focused on the USA display a disproportionately 
         high total Altmetric score (measuring press and public dissemination of journal articles) relative 
         to its comparatively low vulnerability score. This suggests that top-tier journals’ climate change 
         publications focused on the USA potentially generate more public interest than studies focused on 
         other countries, and that the vulnerability of a country may not, in and of itself, command major 
         public interest. LDC countries in orange form a cingular cluster towards the right, showing
         high values of vulnerability index (ND-GAIN) and systematically low values of total Altmetric Scores, 
         further showcasing the lack of coverage these articles receive. """

# ------------------------------------------------------------------------------------------------------
# Region-wise Overall Author Count
# ------------------------------------------------------------------------------------------------------

region_wise_author_working = """The bar plot shows total count of authors of research articles
          for each UNFCCC Grouping, along with special case studies for USA, China and India.
          The total count of authors is calculated by counting all authors (including
          corresponding authors) of each research article and aggregate it over UNFCCC group.
          \n**Interact with the plot:** Users can use the hover functionality over any bar to 
          get all information about the particular country or group."""
          
region_wise_author_takeaway = """In addition to investigating what countries and groupings serve as the focus 
          of high-impact climate change research, it is also important to examine who is contributing 
          this research. We attempt this by evaluating author affiliation by country, using the bibliographic 
          information for each of the studies in our dataset. Most authors on the high-impact studies in our 
          dataset are affiliated with institutions based in Annex-1 countries, ~75% of our dataset, rather 
          than nonAnnex1. These results are also strongly significant (p =2.47e-08), with a relatively 
          large effect size (0.67) and high power (0.93). Again, the USA alone has more authors than all 
          of the nonAnnex-1 countries combined. China also displays a large number of authors, more so 
          than the LDC grouping.  """ 
          
# ------------------------------------------------------------------------------------------------------
# Year-wise Overall Author Count
# ------------------------------------------------------------------------------------------------------
         
year_wise_author_working = """The line chart shows the total number of authors per group over our entire 
          timeframe (2011-2023). USA, China and India are represented by dash lines as special case studies
          of these groups. For a given marker, the 'Total Authors' refers to the sum total number of authors
          of each the research articles published on countries belonging to a UNFCCC group for the current year.
          \n**Interact with the plot:** Users can use the hover functionality over any marker to 
          get all information about the group at that given year. The crop functionality can also be used to
          zoom into any portion of the graph. To zoom-out, double-click anywhere on the graph."""

year_wise_author_takeaway = """Authors of these studies overwhelmingly hail from Annex-1 countries, although 
          there has been a marked increase in authors from nonAnnex-1 countries since 2020. This may be 
          largely driven by authors affiliated with Chinese institutions, which also display similar sharp 
          increases over this period. While publications from authors affiliated with USA institutions are 
          also rising, a marked decline is present from ~2020-2022. Authors affiliated with Indian 
          institutions are also increasing at the end of the timeseries, albeit modestly, while  
          authors from LDC countries appear to have declined in 2022. """
          
# ------------------------------------------------------------------------------------------------------
# Heatmap for Authors
# ------------------------------------------------------------------------------------------------------
         
heatmap_author_working = """This chart represents a correlation matrix to indicate who published on what.
          The horizontal axis represents the Author Groups, i.e. which region does the author belong to. 
          This information is pulled from each author's bibliographic information and mapping their 
          institution to the appropriate country and UNFCCC Grouping. The vertical axis represents the 
          Research Groups, i.e. which group was the research in question based on. The correlation count is
          displayed as the total number of interactions between Author and Research Groups. For example, a 
          research article based on USA could have 5 authors from Annex-1 country and 2 from nonAnnex-1 
          countries, and these counts would respectively add to (Annex-1, USA) and (nonAnnex-1, USA).
          \n**Interact with the plot:** Hovering over each cell shows the cross-section of (Author Region, 
          Research Region). """

heatmap_author_takeaway = """To address the specific question of “who is focusing on where”, we compare 
          country/grouping of focus and author affiliation. Authors affiliated with Annex-1 institutions 
          appear to publish a similar number of studies focused on Annex-1 and nonAnnex-1 countries. 
          Annex-1 authors also publish a similar number of studies on the USA as they do on LDC countries. 
          In contrast, nonAnnex-1 affiliated authors in our dataset publish more on nonAnnex-1 countries 
          relative to other country groupings. Fewer nonAnnex-1 authors are publishing on other countries 
          and groupings, and a comparable number are publishing on Annex-1 regions and China. USA authors 
          publish a similar number of studies on Annex-1 and nonAnnex-1 countries, and then subsequently 
          the next most frequent domain of focus is the USA followed by LDCs. Most Chinese affiliated 
          authors publish on nonAnnex-1 countries and in particular on China, and likewise most Indian 
          authors publish on India. LDC authors primarily publish on LDC countries and nonAnnex-1 countries 
          (which we note all LDCs are also nonAnnex-1).  
          \nIt is also interesting to note that when we look at nonAnnex-1 studies, the Annex-1 authors and 
          specifically USA-affiliated authors dominate the publications in comparison to authors affiliated 
          with nonAnnex-1 group themselves. Similar trend can be seen for LDC group publications as well."""
         
         
         
         
         
         
         
         
         
         
