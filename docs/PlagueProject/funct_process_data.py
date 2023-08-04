# Python 3.11.2
# Import packages

from shapely import wkt
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import Point
import scipy.integrate as scipy
import scipy.optimize as optimize
import scipy.stats as stats
from thefuzz import process
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
import pylab as pl
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import json  # for pretty printing
import geopandas as gpd
import os
import re
import folium
from mapclassify import classify
import math
from datetime import datetime, timedelta
import math
import networkx as nx # for network analysis, graphs
import plotly.express as px

# Function to get all parishes from a specific region
def parishesByregion(df: pd.DataFrame, region: str) -> pd.DataFrame:
    return df.loc[df['Region'] == region]

def get_Names(data: pd.DataFrame, heading: str) -> list:
    return data[heading]

def replace_strings_and_regex(dataframe: pd.DataFrame, heading: str, patterns):
    for pattern in patterns:
        dataframe.loc[:, heading] = dataframe.loc[:, heading].apply(
            lambda x: re.sub(pattern, '', x))
    # print(dataframe.loc[:, heading].__dict__)
    return dataframe

def process_dataframe_rep(df, groupby_column, year_column):
    # Group a Pandas DataFrame by a column
    parish_grp = df.groupby([groupby_column])
    # Get the unique values of a column as a list
    parish_grp_name = parish_grp[groupby_column].unique().tolist()
    result_df = pd.DataFrame()
    for name in parish_grp_name:
        grp_name = parish_grp.get_group(name[0])
        result_df = pd.concat(
            [result_df, (grp_name[grp_name[year_column] == grp_name[year_column].min()])], axis=0)
    return result_df

def process_dataframe(df, column1: str, column2: str):
    aux_dict = {}
    for i in range(len(df)):
        name_i = df[column1].iloc[i]
        ar_i = df[column2].iloc[i]
        if name_i in aux_dict:
            if ar_i < aux_dict[name_i]['min']:
                aux_dict[name_i] = {'min': ar_i, 'position': i}
        else:
            aux_dict[name_i] = {'min': ar_i, 'position': i}
    final_positions = [value['position'] for key, value in aux_dict.items()]
    return df.iloc[final_positions]

def check_name(data: pd.DataFrame, heading: str, name: str):
    filt_name = data[heading].str.contains(name, na=False)
    return data.loc[filt_name]

def is_similar(name1, name2, threshold=0.8):
    max_len = max(len(name1), len(name2))
    return levenshtein_distance(name1, name2) / max_len < threshold

def fuzzy_match(
    df_left, df_right, column_left, column_right, threshold=90, limit=1
):
    # Create a series
    series_matches = df_left[column_left].apply(
        # Creates a series with id from df_left and column name _column_left_, with _limit_ matches per item
        lambda x: process.extract(x, df_right[column_right], limit=limit)
    )

    # Convert matches to a tidy dataframe
    df_matches = series_matches.to_frame()
    # Convert list of matches to rows
    df_matches = df_matches.explode(column_left)
    df_matches[
        ['match_string', 'match_score', 'df_right_id']
    ] = pd.DataFrame(df_matches[column_left].tolist(), index=df_matches.index)       # Convert match tuple to columns
    # Drop column of match tuples
    df_matches.drop(column_left, axis=1, inplace=True)

    # Reset index, as in creating a tidy dataframe we've introduced multiple rows per id, so that no longer functions well as the index
    if df_matches.index.name:
        index_name = df_matches.index.name     # Stash index name
    else:
        index_name = 'index'        # Default used by pandas
    df_matches.reset_index(inplace=True)
    # The previous index has now become a column: rename for ease of reference
    df_matches.rename(columns={index_name: 'df_left_id'}, inplace=True)

    # Drop matches below threshold
    df_matches.drop(
        df_matches.loc[df_matches['match_score'] < threshold].index,
        inplace=True
    )

    return df_matches

# Adding geographical characteristics to the data

def get_area(gpd: gpd.GeoDataFrame, heading: str = 'geometry'):
    for i in range(len(gpd)):
        gpd['area_m2'] = shape(gpd.loc[i][heading]).area
        gpd['area_km2'] = gpd['area_m2']/1000000
    return gpd

def get_centroid(gpd: gpd.GeoDataFrame):
    for i in range(len(gpd)):
        gpd.loc[i, 'centroid'] = gpd.geometry.centroid[i]
    return gpd

def distance_btw_centroids(gpd: gpd.GeoDataFrame):
    for i in range(len(gpd)):
        gpd.loc[i, 'distance'] = gpd.geometry.distance(gpd.centroid[i])
    return gpd

# Computing the distance between the centroids and shared borders

def compute_info(gdp: gpd.GeoDataFrame,
                 column_name: str,
                 column_geometry: str = 'geometry',
                 column_centroid: str = 'centroid',
                 units: int = 1) -> dict:

    nPolygons = len(gdp)
    info = defaultdict(dict)

    for i in range(nPolygons):
        polygon_i = gdp.iloc[i][column_geometry]
        centroid_i = gdp.loc[i, column_centroid]
        name_i = gdp.iloc[i][column_name]

        for j in range(i+1, nPolygons):
            polygon_j = gdp.iloc[j][column_geometry]
            centroid_j = gdp.loc[j, column_centroid]
            name_j = gdp.iloc[j][column_name]

            distance = centroid_i.distance(centroid_j) / units * 1.0
            info["distance"][(i, j)] = distance  # in meters
            info["distance"][(j, i)] = distance  # in meters
            info["distance"][(name_i, name_j)] = distance  # in meters
            info["distance"][(name_j, name_i)] = distance  # in meters

            shared_border = polygon_i.intersection(polygon_j)
            info["shared_border"][(
                i, j)] = shared_border.length if shared_border != None else 0  # in meters
            info["shared_border"][(
                j, i)] = shared_border.length if shared_border != None else 0  # in meters
            info["shared_border"][(name_i, name_j)
                                  ] = info["shared_border"][(i, j)]  # in meters
            info["shared_border"][(name_j, name_i)
                                  ] = info["shared_border"][(j, i)]  # in meters

    return info

# Assigning colors (red -> Plague and blue -> NoPlague)
def colorByColumn(gpd: gpd.GeoDataFrame, heading: str = 'BeginPlaguePeriod'):
    gpd['color'] = gpd[heading].map(lambda x: 'blue' if pd.isna(x) else 'red')
    pass

def begin_days_between(d1, d2):
    if (type(d1) == float and math.isnan(d1)) or \
       (type(d2) == float and math.isnan(d2)):
        return None
    return abs((d2 - d1).days)

def end_days_between(d1, d2):
    if (type(d1) == float and math.isnan(d1)) or (type(d2) == float and math.isnan(d2)):
        return None
    # Create first day of the first month
    first_day_d1 = datetime(d1.year, d1.month, 1)
    # Create last day of the second month
    if d2.month == 12:
        last_day_d2 = datetime(d2.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day_d2 = datetime(d2.year, d2.month + 1, 1) - timedelta(days=1)

    return abs((last_day_d2 - first_day_d1).days)

# Defining the seasonal function

def gaussian(x, media, std):
    return np.exp(-((x - media) ** 2) / (2 * std ** 2))

def seasonal_transmission_rate(t, bump_center, bump_width, bump_height):
    return bump_height * gaussian(t % 365, bump_center, bump_width) + bump_height * gaussian(t % 365 - 365, bump_center, bump_width) + bump_height * gaussian(t % 365 + 365, bump_center, bump_width)

def transmission_matrix_beta(gdf: gpd.GeoDataFrame, column_name: str = 'ParishName'):
    same_names = gdf[column_name].apply(lambda x: gdf[column_name] == x).values
    beta_matrix = np.zeros_like(same_names, dtype=float)
    np.fill_diagonal(beta_matrix, 1)
    return beta_matrix

# def transmission_matrix_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', column_centroid: str = 'centroid', column_pop: str = 'BEF1699', column_name: str = 'ParishName'):
    
#     # Calculate distances between all centroids
#     centroid_distances = gdf[column_centroid].apply(
#         lambda x: gdf[column_centroid].apply(lambda y: x.distance(y))).values
    
#     # Calculate population products for all pairs of polygons
#     pop_products = np.outer(gdf[column_pop], gdf[column_pop])

#     # Create a boolean matrix to identify intersecting polygons
#     intersecting_polygons = gdf[column_geometry].apply(
#         lambda x: gdf[column_geometry].intersects(x)).values
    
#     # Create a boolean matrix to identify same names
#     same_names = gdf[column_name].apply(lambda x: gdf[column_name] == x).values

#     # For non-intersecting polygons with the same name, set the distance to infinity
#     centroid_distances[np.logical_or(~intersecting_polygons, same_names)] = np.inf

#     # For polygons with centroid distance less than or equal to 5km
#     # Calculate the population product divided by the square of the distance

#     less_distance = centroid_distances <= 5000
#     centroid_distances[less_distance] = pop_products[less_distance] / (centroid_distances[less_distance]**2)

#     # Replace diagonal elements in centroid_distances with 1 to avoid division by
#     np.fill_diagonal(centroid_distances, 1)

#     # Calculate the transmission matrix
#     p_matrix = (pop_products / (centroid_distances**2))
#     np.fill_diagonal(p_matrix, 0)
#     return p_matrix 

def transmission_matrix_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', column_centroid: str = 'centroid'):
    # Initialize an empty matrix of size n x n (where n is number of polygons)
    n = len(gdf)
    matrix = np.zeros((n, n))

    # Loop through each pair of polygons
    for i in range(n):
        for j in range(i+1, n):  # start from i+1 to avoid redundant calculations
            # Get the centroid of each polygon from the GeoDataFrame
            centroid_i = gdf.loc[i, column_centroid]
            centroid_j = gdf.loc[j, column_centroid]
            # Calculate the distance between the centroids in kilometers
            distance = centroid_i.distance(centroid_j)/1000
            # If polygon i intersects polygon j,
            # set matrix[i][j] and matrix[j][i] to 1
            if gdf.iloc[i][column_geometry].intersects(gdf.iloc[j][column_geometry]):
                matrix[i][j] = matrix[j][i] = 1  # set both matrix[i][j] and matrix[j][i] to 1
            # If polygon i does not intersect polygon j, 
            # check that the distance between their centroids is <= 10km
            elif distance <= 5:
                matrix[i][j] = matrix[j][i] = 1  # set both matrix[i][j] and matrix[j][i] to 1

    return matrix

# def transmission_matrix_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', column_centroid: str = 'centroid', column_pop: str = 'BEF1699', column_name: str = 'ParishName'):
#     # Initialize an empty matrix of size n x n (where n is number of polygons)
#     same_names = gdf[column_name].apply(lambda x: gdf[column_name] == x).values
#     matrix = np.zeros_like(same_names, dtype=float)
#     np.fill_diagonal(matrix, 1)
#     n = len(same_names)    
#     # Loop through each pair of polygons
#     for i in range(n):
#         for j in range(i+1, n):  # start from i+1 to avoid redundant calculations
#             # Get the name of each polygon from the GeoDataFrame
#             name_i = gdf.loc[i, column_name]
#             name_j = gdf.loc[j, column_name]
#             # Get the centroid of each polygon from the GeoDataFrame
#             centroid_i = gdf.loc[i, column_centroid]
#             centroid_j = gdf.loc[j, column_centroid]
#             # Get the population of each polygon from the GeoDataFrame
#             pop_i = gdf.loc[i, column_pop]
#             pop_j = gdf.loc[j, column_pop]
#             # Calculate the distance between the centroids in kilometers
#             distance = centroid_i.distance(centroid_j)/1000
#             # Calculate the population product
#             pop_product = pop_i * pop_j
#             # If polygon i intersects polygon j,
#             # set matrix[i][j] and matrix[j][i] to 1
#             if gdf.iloc[i][column_geometry].intersects(gdf.iloc[j][column_geometry]) and name_i != name_j:
#                 matrix[i][j] = matrix[j][i] = pop_product/distance  
#             # If name_i is equal to name_j, 
#             # check that the distance between their centroids is <= 10km
#             elif name_i == name_j:
#                 matrix[i][j] = matrix[j][i] = np.inf
#             # If polygon i does not intersect polygon j, 
#             # check that the distance between their centroids is <= 10km
#             elif distance <= 10:
#                 matrix[i][j] = matrix[j][i] = pop_product/(distance**2)  # set both matrix[i][j] and matrix[j][i] to 1
#     np.fill_diagonal(matrix, 0)    
#     return matrix

# def transmission_matrix_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', 
#                           column_centroid: str = 'centroid', column_pop: str = 'BEF1699', 
#                           column_name: str = 'ParishName'):
#     # Initialize an empty matrix of size n x n (where n is number of polygons)
#     n = len(gdf)
#     matrix = np.zeros((n, n), dtype=float)

#     # Loop through each pair of polygons
#     for i in range(n):
#         for j in range(i+1, n):  # start from i+1 to avoid redundant calculations
#             # Get the name of each polygon from the GeoDataFrame
#             name_i = gdf.loc[i, column_name]
#             name_j = gdf.loc[j, column_name]

#             # Skip if both polygons are the same
#             if name_i == name_j:
#                 continue

#             # Get the centroid of each polygon from the GeoDataFrame
#             centroid_i = gdf.loc[i, column_centroid]
#             centroid_j = gdf.loc[j, column_centroid]

#             # Get the population of each polygon from the GeoDataFrame
#             pop_i = gdf.loc[i, column_pop]
#             pop_j = gdf.loc[j, column_pop]

#             # Calculate the distance between the centroids in meters
#             distance = centroid_i.distance(centroid_j)/1000

#             # If polygon i intersects polygon j or the distance between their centroids is <= 10km,
#             # calculate the population product divided by the square of the distance
#             if gdf.iloc[i][column_geometry].intersects(gdf.iloc[j][column_geometry]):
#                 matrix[i][j] = matrix[j][i] = (pop_i * pop_j) / (distance ** 2 if distance > 0 else 1)
#             elif distance <= 10:
#                 matrix[i][j] = matrix[j][i] = (pop_i * pop_j) / (distance ** 2 if distance > 0 else 1)

#     return matrix
