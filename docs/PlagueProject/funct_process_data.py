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
import networkx as nx # for network analysis, graphs
import plotly.express as px
from skopt import gp_minimize # for Bayesian optimization
from pandas.tseries.offsets import DateOffset, MonthEnd


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
                 column_name: str = 'ParishName',
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

# Assigning colors (red -> Plague and blue -> NoPlague)
def classByPlague(gpd: gpd.GeoDataFrame, heading: str = 'BeginPlaguePeriod'):
    gpd['plague'] = gpd[heading].map(lambda x: 0 if pd.isna(x) else 1)
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

def convert_to_int(row, death_col : str ='VictimsNumber'):
    if pd.notna(row[death_col]):
        return int(row[death_col])
    else:
        return None

def sort_by_date(gdf, column_date: str = 'new_format_BeginPlaguePeriod'):
    gdf_copy = gdf.copy()
    gdf_copy.sort_values(by=[column_date],   # Row or columns names to sort by
                    axis=0,       # Sort Rows axis = 0
                    ascending=True,  # Sort ascending or descending?
                    inplace=True     # Modify the DataFrame in place (do not create a new object)
                    )
    gdf_copy.reset_index(drop=True, inplace=True)
    return gdf_copy

def add_Begin_End_days(gdf, begin_column:str = 'new_format_BeginPlaguePeriod', end_column:str = 'new_format_EndPlaguePeriod'):
    gdf_copy = gdf.copy()
    # Create a new column called "BeginDaysPlague"
    gdf_copy["BeginDaysPlague"] = gdf.apply(lambda row: begin_days_between(gdf[begin_column].iloc[0]
                                                                         ,row[begin_column])
                                                                         , axis=1  # axis = 1 means apply function to each row
    )
    
    # Create a new column called "EndDaysPlague"
    gdf_copy['EndDaysPlague'] = gdf.apply(lambda row: end_days_between(gdf[begin_column].iloc[0]
                                            , row[end_column]) if pd.notna(row[end_column]) else None
                                            , axis=1)

    # Replace NaN values with a value in some columns (e.g., 0)
    gdf_copy['BeginDaysPlague'].fillna(0, inplace=True)
    gdf_copy['EndDaysPlague'].fillna(0, inplace=True)
    #gdf_copy[death_column].fillna(None, inplace=True)
    
    # Changing the type of some columns from float to integer for the optimization process
    gdf_copy['BeginDaysPlague'] = gdf_copy['BeginDaysPlague'].astype(int)
    gdf_copy['EndDaysPlague'] = gdf_copy['EndDaysPlague'].astype(int)
    #gdf_copy[death_column] = gdf_copy['VictimsNumber'].astype(int)
        
    gdf_copy.reset_index(drop=True, inplace=True)
    return gdf_copy

# Function to call the data from the excel files
def get_parish_data(parish_name, parish_folder):
    parish_path = os.path.join(parish_folder, parish_name + '.xlsx')
    parish = pd.read_excel(parish_path, sheet_name='Plague')

    # Convert 'EndDate' to datetime with appropriate format
    parish['NewEndDate'] = pd.to_datetime(parish['EndDate'], format='%b %Y')
    parish['NewEndDate'] = parish['NewEndDate'].dt.to_period('M')
    parish['first_day'] = parish['NewEndDate'].dt.to_timestamp()
    parish['last_day'] = parish['NewEndDate'].dt.to_timestamp(how='end')

    # Add a column with the days since the first date and then cumsum
    parish['Days'] = parish['last_day'].dt.daysinmonth
    parish['Days'] = parish['Days'].cumsum()
    return parish

# Function to get the population of a specific parish
def get_parish_info(parish_name, df: pd.DataFrame, column_name='ParishName', column_pop='BEF1699'):
    pop_df = df[(df[column_name] == parish_name)][column_pop]
    name_df = df[(df[column_name] == parish_name)][column_name]
    
    if not pop_df.empty and not name_df.empty:
        pop_parish = pop_df.values[0]
        name_parish = name_df.values[0]
    else:
        pop_parish = None
        name_parish = None

    return pop_parish, name_parish

# Defining the seasonal function

def gaussian(x, media, std):
    return np.exp(-((x - media) ** 2) / (2 * std ** 2))

def seasonal_transmission_rate(t, bump_center, bump_width, bump_height):
    return bump_height * gaussian(t % 365, bump_center, bump_width) + bump_height * gaussian(t % 365 - 365, bump_center, bump_width) + bump_height * gaussian(t % 365 + 365, bump_center, bump_width)

def transmission_matrix_beta(gdf: gpd.GeoDataFrame, beta:np.array, column_name: str = 'ParishName'):
    same_names = gdf[column_name].apply(lambda x: gdf[column_name] == x).values
    beta_matrix = np.zeros_like(same_names, dtype=float)
    np.fill_diagonal(beta_matrix, beta) 
    return beta_matrix

# def transmission_matrix_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', column_centroid: str = 'centroid', column_pop: str = 'BEF1699', column_name: str = 'ParishName'):
    
#     # Calculate distances between all centroids in meters
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

#     # Replace diagonal elements in centroid_distances with 1 to avoid division by
#     np.fill_diagonal(centroid_distances, 1)

#     # Calculate the transmission matrix
#     p_matrix = (pop_products / (centroid_distances**2))
#     np.fill_diagonal(p_matrix, 0)
#     return p_matrix 

# Definition without checking if the polygons intersect
# def transmission_matrix2_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', column_centroid: str = 'centroid', column_pop: str = 'BEF1699', column_name: str = 'ParishName'):
    
#     # Calculate distances between all centroids in meters
#     centroid_distances = gdf[column_centroid].apply(
#         lambda x: gdf[column_centroid].apply(lambda y: x.distance(y))).values
    
#     # Calculate population products for all pairs of polygons
#     pop_products = np.outer(gdf[column_pop], gdf[column_pop])

#     # Create a boolean matrix to identify same names
#     same_names = gdf[column_name].apply(lambda x: gdf[column_name] == x).values

#     # For polygons with the same name, set the distance to infinity
#     centroid_distances[same_names] = np.inf

#     # Replace diagonal elements in centroid_distances with 1 to avoid division by zero
#     np.fill_diagonal(centroid_distances, 1)

#     # Calculate the transmission matrix
#     p_weight = (pop_products / (centroid_distances**2))
#     np.fill_diagonal(p_weight, 0)
#     return p_weight

# def transmission_matrix_beta(gdf: gpd.GeoDataFrame, beta:np.array, column_name: str = 'ParishName'):
#     unique_names = gdf[column_name].unique()
#     len_unique_names = len(unique_names)
#     beta_matrix = np.zeros((len_unique_names,len_unique_names), dtype=float)
#     np.fill_diagonal(beta_matrix, beta) 
#     return beta_matrix

# # Transmission matrix defined for SCENARIO 1. 
# def trans_matrix1(gdf: gpd.GeoDataFrame, beta:float, p:float, column_name: str = 'ParishName', column_geometry: str = 'geometry'):
#     unique_names = gdf[column_name].unique()
#     len_unique_names = len(unique_names)
#     trans_matrix = np.full((len_unique_names,len_unique_names),p, dtype=float)
#     for i in range(len_unique_names):
#         for j in range(i+1, len_unique_names): 
#             polygon_i = gdf[gdf[column_name] == unique_names[i]][column_geometry].values[0]
#             polygon_j = gdf[gdf[column_name] == unique_names[j]][column_geometry].values[0]
#             # If polygons don't touch, set value in trans_matrix to 0
#             if not polygon_i.touches(polygon_j):
#                 trans_matrix[i,j] = 0
#                 trans_matrix[j,i] = 0
#     np.fill_diagonal(trans_matrix, beta) 
#     return trans_matrix

# # Transmission matrix defined for SCENARIO 2 and SCENARIO 3. 
# def trans_matrix2(gdf: gpd.GeoDataFrame, beta, p:float, column_name: str = 'ParishName', column_geometry: str = 'geometry'):
#     unique_names = gdf[column_name].unique()
#     len_unique_names = len(unique_names)
#     trans_matrix = np.full((len_unique_names,len_unique_names),p, dtype=float)
#     for i in range(len_unique_names):
#         for j in range(i, len_unique_names): 
#             if i != j:
#                 polygon_i = gdf[gdf[column_name] == unique_names[i]][column_geometry].values[0]
#                 polygon_j = gdf[gdf[column_name] == unique_names[j]][column_geometry].values[0]
#                 # If polygons don't touch, set value in trans_matrix to 0
#                 if not polygon_i.touches(polygon_j):
#                     trans_matrix[i,j] = 0
#                     trans_matrix[j,i] = 0
#             else:
#                 trans_matrix[i,j] = beta[i]
#     return trans_matrix

def getValueAt(array, n, i, j):
    if i == j: return 0
    if i < j:
        return array[int(j*(j-1)/2) + i]
    return getValueAt(array, n, j, i)

# Transmission matrix defined for SCENARIO 4.
def trans_matrix4(gdf: gpd.GeoDataFrame, beta:np.array, p:np.array, n, column_name: str = 'ParishName', column_geometry: str = 'geometry'):
    # Get unique parish names 
    unique_names = gdf[column_name].unique()
    len_unique_names = len(unique_names)

    # Initialize the beta matrix
    beta_matrix = np.zeros((len_unique_names, len_unique_names), dtype=float)
    np.fill_diagonal(beta_matrix, beta)

    # Initialize the transmission matrix between patches
    trans_matrix = np.full((len_unique_names, len_unique_names), 0.0, dtype=float)

    for i in range(len_unique_names):
        for j in range(i+1,len_unique_names):
            name_i = unique_names[i]
            name_j = unique_names[j]
            polygon_i = gdf[gdf[column_name] == name_i][column_geometry].values[0]
            polygon_j = gdf[gdf[column_name] == name_j][column_geometry].values[0]
            pVal = getValueAt(p, n, i, j)

            if polygon_i.touches(polygon_j) and name_i != name_j:
                trans_matrix[i,j] = pVal
                trans_matrix[j,i] = trans_matrix[i,j]
            else:
                trans_matrix[i,j] = 0
                trans_matrix[j,i] = 0

    return beta_matrix + trans_matrix  

# def create_symmetric_matrix(array, n):
#     # Create an empty matrix
#     matrixA = np.zeros((n,n))
    
#     # Fill the upper triangular part of the matrix using advanced indexing
#     for i in range(n-1):
#         matrixA[i+1, :i+1] = array[int(i*(i+1)/2) : int(i*(i+1)/2) + (i + 1)]
#     # Make the matrix symmetric by adding it to its transpose and subtracting the diagonal
#     npSymMatrixA = matrixA + matrixA.T - np.diag(matrixA.diagonal())    
#     return npSymMatrixA

# # p = create_symmetric_matrix(np.array([1,2,3,4,5,6]), 4)
# # print(p)

# # def getValueAt(array,n, i, j):
# #     if i == j: return 0
# #     if i < j:
# #         return array[int(j*(j-1)/2) + i]
# #     return getValueAt(array, n, j, i)

# print("=====")
# # print the matrix using getValueAt function
# for i in range(4):
#     for j in range(4):
#         print(getValueAt([1,2,3,4,5,6], 4, i, j), end=' ')
#     print()


def total_transmission_matrix(gdf: gpd.GeoDataFrame, beta:np.array, p_coeff: np.array, n, column_geometry: str = 'geometry', 
                           column_centroid: str = 'centroid', column_pop: str = 'BEF1699', 
                           column_name: str = 'ParishName'):

    # Get unique parish names 
    unique_names = gdf[column_name].unique()
    len_unique_names = len(unique_names)

    # Initialize the beta matrix
    beta_matrix = np.zeros((len_unique_names, len_unique_names), dtype=float)
    np.fill_diagonal(beta_matrix, beta)

    # Initialize the gravitational matrix
    gravitational = np.full((len_unique_names, len_unique_names), 0.0)

    for i in range(len_unique_names):
        for j in range(i+1,len_unique_names):
            name_i = unique_names[i]
            name_j = unique_names[j]
            centroid_i = gdf[gdf[column_name] == name_i][column_centroid].values[0]
            centroid_j = gdf[gdf[column_name] == name_j][column_centroid].values[0]
            pop_i = gdf[gdf[column_name] == name_i][column_pop].values[0]
            pop_j = gdf[gdf[column_name] == name_j][column_pop].values[0]
            pVal = getValueAt(p_coeff, n, i, j)

            if name_i != name_j:
                gravitational[i,j] = pVal*((pop_i * pop_j) / (centroid_i.distance(centroid_j)**2))
                gravitational[j,i] = gravitational[i,j]
            else:
                gravitational[i,j] = 0
                gravitational[j,i] = 0

    return  beta_matrix + gravitational  

def transmission_matrix2_p(gdf: gpd.GeoDataFrame, p_coeff:np.array, column_geometry: str = 'geometry', 
                           column_centroid: str = 'centroid', column_pop: str = 'BEF1699', 
                           column_name: str = 'ParishName'):

    # Get unique parish names 
    unique_names = gdf[column_name].unique()
    len_unique_names = len(unique_names)

    # Initialize the p_coeff matrix
    p_coeff = np.full((len_unique_names, len_unique_names), p_coeff)

    # Initialize the p_matrix
    p_matrix = np.full((len_unique_names, len_unique_names), 0.0)

    # Initialize the gravitational matrix
    gravitational = np.full((len_unique_names, len_unique_names), 0.0)

    for i in range(len_unique_names):
        for j in range(i+1,len_unique_names):
            name_i = unique_names[i]
            name_j = unique_names[j]
            centroid_i = gdf[gdf[column_name] == name_i][column_centroid].values[0]
            centroid_j = gdf[gdf[column_name] == name_j][column_centroid].values[0]
            pop_i = gdf[gdf[column_name] == name_i][column_pop].values[0]
            pop_j = gdf[gdf[column_name] == name_j][column_pop].values[0]
            if name_i != name_j:
                gravitational[i,j] = (pop_i * pop_j) / (centroid_i.distance(centroid_j)**2)
                gravitational[j,i] = gravitational[i,j]
            else:
                gravitational[i,j] = 0
                gravitational[j,i] = 0
    p_matrix = p_coeff * gravitational   

    return p_matrix 

# def transmission_matrix2_p(gdf: gpd.GeoDataFrame, column_geometry: str = 'geometry', 
#                            column_centroid: str = 'centroid', column_pop: str = 'BEF1699', 
#                            column_name: str = 'ParishName'):

#     # Get unique parish names and create a mapping to indices
#     unique_names = gdf[column_name].unique()
#     name_to_index = {name: index for index, name in enumerate(unique_names)}

#     # Calculate distances between all centroids in meters
#     centroid_distances = np.zeros((len(unique_names), len(unique_names)))
#     for name1, index1 in name_to_index.items():
#         for name2, index2 in name_to_index.items():
#             if name1 != name2:
#                 centroid1 = gdf[gdf[column_name] == name1][column_centroid].values[0]
#                 centroid2 = gdf[gdf[column_name] == name2][column_centroid].values[0]
#                 centroid_distances[index1, index2] = centroid1.distance(centroid2)
#     # Set diagonal elements to infinity to avoid division by zero later
#     np.fill_diagonal(centroid_distances, np.inf)

#     # Calculate population products for all pairs of polygons
#     pop_products = np.zeros((len(unique_names), len(unique_names)))
#     for name1, index1 in name_to_index.items():
#         for name2, index2 in name_to_index.items():
#             pop1 = gdf[gdf[column_name] == name1][column_pop].values[0]
#             pop2 = gdf[gdf[column_name] == name2][column_pop].values[0]
#             pop_products[index1, index2] = pop1 * pop2

#     # Calculate the transmission matrix
#     p_weight = (pop_products / (centroid_distances**2))
#     return p_weight


# def get_parish_data(parish_name, parish_folder):
#     parish_path = os.path.join(parish_folder, parish_name + '.xlsx')
#     parish = pd.read_excel(parish_path, sheet_name='Plague')

#     # Rename two columns
#     parish = parish.rename(columns={'CumDeaths': 'VictimsNumber', 'EndDate': 'EndPlaguePeriod'})
    
#     # Convert 'EndPlaguePeriod' to datetime with appropriate format
#     parish['NewEndDate'] = pd.to_datetime(parish['EndPlaguePeriod'], format='%b %Y')
#     parish['NewEndDate'] = parish['NewEndDate'].dt.to_period('M')
#     parish['first_day'] = parish['NewEndDate'].dt.to_timestamp()
#     parish['last_day'] = parish['NewEndDate'].dt.to_timestamp(how='end')

#     # Add a column with the days since the first date and then cumsum
#     parish['EndDaysPlague'] = parish['last_day'].dt.daysinmonth
#     parish['EndDaysPlague'] = parish['EndDaysPlague'].cumsum()
#     return parish

# # Create a dictionary
# parish_file_dict ={ "YSTAD": get_parish_data('Ystad', southeast_parishes_folder)
#                    , "SKÅRBY": get_parish_data('Skarby', southeast_parishes_folder)
#                    , "SNÅRESTAD": get_parish_data('Snarestad', southeast_parishes_folder)
#                    , "ÖVED": get_parish_data('Oved', middle_parishes_folder)
#                    , "SÖDRA ÅSUM": get_parish_data('SodraAsum', middle_parishes_folder)
#                    , "BARSEBÄCK": get_parish_data('Barseback', southwest_parishes_folder)
#                    , "LILLA BEDDINGE": get_parish_data('LillaBeddinge', southwest_parishes_folder)
#                    , "RÄNG": get_parish_data('Rang', southwest_parishes_folder)
#                    , "SVENSTORP": get_parish_data('Svenstorp', southwest_parishes_folder)
#                    }


# dict_parish ={}
# grouped_by_parish = example.groupby('ParishName')
# group_dict = {}
# for name, data in grouped_by_parish:
#     group_dict[name] = data

# errors = np.zeros(n)
# for i in range(n):
#     current_parish = model_input.patchNames()[i]
#     current_df = group_dict[current_parish]
#     len_data_parish = len(current_df)
#     #print(current_parish, current_df, len_data_parish)
#     # If we only have one data point, we can't calculate the error
#     if len_data_parish < 2:         
#         print(current_parish)    
#         initial_position = current_df['BeginDaysPlague'].values[0]
#         final_position = current_df['EndDaysPlague'].values[0]
#         deaths = current_df['VictimsNumber'].values[0]
#         if (deaths != 0 and final_position != 0):
#             errors[i] = ((initial_position - 1.0)**2 + (final_position - deaths)**2)
#         else:
#             errors[i] = ((initial_position - 1.0)**2)
#         print(current_parish, initial_position, final_position, deaths,errors[i])
#     else:
#         print(current_parish + ' has more than one data point')
#         point_error = 0
#         for j in range(len(data_by_parish.get_group(current_parish))):
#             position = current_df['BeginDaysPlague'].values[j]
#             monthly_deaths = current_df['VictimsNumber'].values[j]
#             point_error = (position - monthly_deaths)**2
#             errors[i] = errors[i] + point_error
#         print(current_parish, errors[i])

# # Calculate the total error
# totalError = np.sum(errors)
# print(totalError)

# # Define the objective function to minimize (sum of squared errors) Slow version
# def objectiveFunction(parameters, gdf: gpd.GeoDataFrame, column_name: str = 'ParishName'
#                       , beginTime: str = 'BeginDaysPlague', endTime: str = 'EndDaysPlague'
#                         , deathData: str = 'VictimsNumber'
#                     ):
#     n = model_input.n
#     # Reshape parameters back to their original shapes
#     beta: np.array = parameters[:n].reshape(n,)
#     mu:  np.array = parameters[n:2*n].reshape(n,)

#     # First, we reshape the  p_coeff vector into a lower triangular matrix
#     p_coeff_lower = np.tril(parameters[2*n:].reshape(n, n))
    
#     # Then, we add the transpose to itself, subtracting the diagonal (which was added twice)
#     p_coeff: np.array = p_coeff_lower + p_coeff_lower.T - np.diag(np.diag(p_coeff_lower))

#     model_info = {'model': SEIRD_model,
#                   'init': {
#                       'S': model_input.S0,
#                       'E': model_input.E0,
#                       'I': model_input.I0,
#                       'R': model_input.R0,
#                       'D': model_input.D0,
#                   },
#                   'gdf': example,
#                   # defining the initial values for the model
#                   'beta': beta,
#                   'p_coeff': p_coeff,
#                   'mu': mu,
#                   'gamma': 0.4,
#                   'sigma': 0.17,
#                   'bump_center': 0.0,
#                   'bump_width': 0.0,
#                   'bump_height': 0.0,
#                   'N': model_input.patchPop(),
#                   'n': model_input.n,
#                   'T': model_input.maxDays()}

#     model_sol = generate_sol(model_info)
#     totalError = 0
#     n = model_info['n']

#     # Create a dictionary where the key is the parish name and the value is the dataframe
#     grouped_by_parish = gdf.groupby(column_name)
#     group_dict = {}
#     for name, data in grouped_by_parish:
#         group_dict[name] = data

#     # Calculate the error for each patch
#     errors = np.zeros(n)
        
#     for i in range(n):
#         current_parish = model_input.patchNames()[i]
#         current_df = group_dict[current_parish]
#         len_data_parish = len(current_df)
#         if len_data_parish < 2:         
#             initial_position = current_df[beginTime].values[0]
#             final_position = current_df[endTime].values[0]
#             deaths = current_df[deathData].values[0]
#             if (deaths != 0 and final_position != 0):
#                 try:
#                     errors[i] = ((model_sol['D'][i][initial_position] - 1.0)**2 + (
#                         model_sol['D'][i][final_position] - deaths)**2)
#                 except:
#                     print(
#                         f"Error at: n={n}, i={i}, final_position={final_position}, len(model_sol['D'])= {len(model_sol['D'])}, model_sol['D'][i] = {model_sol['D'][i]}, deathData[i] = {deathData[i]}")
#             else:
#                 errors[i] = ((model_sol['D'][i][initial_position] - 1.0)**2)
#         else:
#             point_error = 0
#             for j in range(len_data_parish):
#                 position = current_df[endTime].values[j]
#                 monthly_deaths = current_df[deathData].values[j]
#                 point_error = (model_sol['D'][i][position] - monthly_deaths)**2
#                 errors[i] = errors[i] + point_error
    
#     # Calculate the total error
#     totalError = np.sum(errors)
#     return totalError

def count_infected_by_month(df, date, n, column_name: str = 'ParishName'
                            , start_date: str = 'BeginPlaguePeriod'
                            , end_date: str = 'EndPlaguePeriod'):
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Convert your date columns to datetime format
    df_copy[start_date] = pd.to_datetime(df_copy[start_date], format='%b %Y')
    df_copy[end_date] = pd.to_datetime(df_copy[end_date], format='%b %Y', errors='coerce')

    # Replace NaT with corresponding date in start_date column plus n months
    df_copy[end_date] = df_copy[end_date].fillna(df_copy[start_date] + DateOffset(months=n))

    # Convert your date to datetime format
    date = pd.to_datetime(date, format='%b %Y')

    # Add the converted date to a new column in df
    df_copy['ConvertedDate'] = date

    # Define the range of dates
    dates = pd.date_range(start=date, end=df_copy[end_date].max(), freq='MS')

    # Create a unique identifier combining Parish and date ranges
    df_copy['UniqueID'] = df_copy[column_name].astype(str) + '_' + df_copy[start_date].astype(str) + '_' + df_copy[end_date].astype(str)

    # Create a dataframe to store the results
    results = pd.DataFrame({'date': dates
                            , 'DaysFromInitialDate': (dates - df_copy[start_date].min()).days
                            , 'NumberInfectedParishes': 0
                            , 'CumInfectParishes': 0
                            , 'EndOfMonth': (dates + MonthEnd(1))
                            })

    # Initialize an empty list to store the sets of infected parishes
    infected_parishes = []

    # Iterate over the dates
    for date in dates:
        # Count nodes where infection start date is before or on the given date 
        # and either there is no end date or the end date is after the given date
        infected_nodes = df_copy[(df_copy[start_date] <= date) & (df_copy[end_date] >= date)]
        
        # Store the results
        results.loc[results['date'] == date, 'NumberInfectedParishes'] = infected_nodes['UniqueID'].nunique()  # Count only unique instances

        # Add the set of infected parishes to the list
        infected_parishes.append(set(infected_nodes[column_name]))

    # Add a new column to count the days from the initial date to the end of the month
    results['DaysToEndOfMonth'] = (results['EndOfMonth'] - df_copy[start_date].min()).dt.days

    # Add a new column with the sets of infected parishes
    results['InfectedParishes'] = infected_parishes  

    # Calculate the cumulative number of infected parishes by month using the sets
    CumInfectParishes = np.zeros(len(dates), dtype=int)
    
    if len(infected_parishes[0]) > 0:
        CumInfectParishes[0] = len(infected_parishes[0])
        # Defining a variable to store the union of the infected parishes
        union_infected_parishes = set(infected_parishes[0])  
    else:
        union_infected_parishes = set()

    for i in range(1, len(infected_parishes)): 
        if len(infected_parishes[i]) > 0: 
            new_infections = infected_parishes[i].difference(union_infected_parishes)
            CumInfectParishes[i] = CumInfectParishes[i-1] + len(new_infections)
            # Update the union of infected parishes
            union_infected_parishes.update(new_infections)
        else:
            CumInfectParishes[i] = CumInfectParishes[i-1]

    # Add a new column with the cumulative number of infected parishes
    results['CumInfectParishes'] = CumInfectParishes    
    return results