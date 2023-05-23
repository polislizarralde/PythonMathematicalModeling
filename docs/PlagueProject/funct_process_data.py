# Python 3.11.2
# Import packages

from shapely.geometry import shape
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
from collections import defaultdict
import json  # for pretty printing
import geopandas as gpd
from shapely import wkt
import os
import re
import folium
from mapclassify import classify
from shapely.geometry import Polygon
from shapely.geometry import Point
import math
from datetime import datetime, timedelta


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
