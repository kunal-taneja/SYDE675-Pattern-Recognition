# SYDE675-Pattern-Recognition

This repo includes all the practical problems learnt and implemented in the grad course at University of Waterloo. With the help of unsupervised learning, areas of Toronto were analysed for describing criminal patterns and safety of the Toronto's neighbourhoods. 

## Toronto Crime Data Analysis


### Introduction

Public safety and protection is the need of the hour in large cities like Toronto. Law enforcement
agencies in large cities have this uphill task of identifying criminal activities, and a lot of resources
and time is wasted in identifying such crime hot spots in the form of surveillance,investigations and
man-hunt. The agencies can work effectively if they have prior knowledge about a neighborhood
and the crime map of such sensitive areas. Toronto police has setup a public safety data portal
where they have listed down historic crime datasets, we are interested in the Major Crime
Indicators Dataset, this dataset includes all Major Crime Indicators (MCI) occurrences by
reported date and related offences from 2014 to 2017.


### Problem Statement

Our study aims at identifying violent and non-violent neighbourhoods in the city of Toronto, while
providing a better visualization for the public. We will be trying to cluster the crime prone areas
with respect to different major crimes that have occurred in the past. The major challenge here
being to understand the versatile data available to us and employ different pattern recognition
techniques to provide a better crime heat map. We intend to perform data analysis to find out
what crimes occurs in what time of the day and the geographic location associated with crime.
These findings would be performed using different clustering techniques learned in class and a
comparative study would be provided.

### Methodology


Identifying crime and predicting dangerous hotspots at a certain time and place could provide
a better visualization for both public and authorities, and aid in terms of proper planning and
safety measure to stop the antisocial activities from happening in the community. Hence, in our
project, we try to model a relationship between several criminal patterns, the behaviour and degree
of crime. Due to the dataset being unlabeled, we aim to apply unsupervised learning techniques
to get an inference about hidden patterns. Upon observing the dataset we encountered a lot of
redundant and irrelevant entries that would need cleaning and reduction.
So, we aim at achieving three major tasks namely:
* **Data Preprocessing** : Including Data cleaning, Scaling, Normalization and Dimensionality
reduction.
* **Clustering Models** : We will group the different demographic locations into either violent
or non-violent categories by implementing at-least two Clustering techniques. Also, we will
try to compare the result of each technique and present them graphically.
* **Data Visualization** : This would include scatter plots, histograms, line graphs describing
different features when plotted with respect to each other. In the end we will generate a heat
map which would represent the demographic locations based on the category to which they
belong. This would help in observing which area has the highest and lowest crime rates.


The goal of this project is visualize the data and successfully categorize the different areas of
Toronto city into two groups or clusters based on the time and location of criminal occurrences
as in the dataset. At the end based on the information learned, we would plot a heatmap which
would graphically superimpose these clusters on the actual map of Toronto City.
