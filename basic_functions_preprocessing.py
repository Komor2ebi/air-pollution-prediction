import pandas as pd

# Basic functions for preprocessing data

def drop_columns(df):
    # Drop specified columns from the dataframe
    columns_to_drop = ['target_min', 'target_max', 'target_count', 'target_variance']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df

def convert_date_and_extract_components(df):
    # Convert 'date' column to datetime dtype using the specified format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # Extract month, day, and day of the week from the 'date' column
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    return df

def drop_zero_rows(df_X, df_y):
    # List of columns to check for 0-values
    columns_with_zeros = [
        'L3_NO2_NO2_column_number_density',
        'L3_NO2_NO2_slant_column_number_density',
        'L3_NO2_absorbing_aerosol_index',
        'L3_NO2_sensor_altitude',
        'L3_NO2_stratospheric_NO2_column_number_density',
        'L3_NO2_tropopause_pressure',
        'L3_NO2_tropospheric_NO2_column_number_density',
        'L3_O3_O3_column_number_density',
        'L3_O3_O3_effective_temperature',
        'L3_CO_CO_column_number_density',
        'L3_CO_H2O_column_number_density',
        'L3_CO_cloud_height',
        'L3_CO_sensor_altitude',
        'L3_HCHO_HCHO_slant_column_number_density',
        'L3_HCHO_tropospheric_HCHO_column_number_density',
        'L3_HCHO_tropospheric_HCHO_column_number_density_amf',
        'L3_SO2_SO2_column_number_density',
        'L3_SO2_SO2_column_number_density_amf',
        'L3_SO2_SO2_slant_column_number_density',
        'L3_SO2_absorbing_aerosol_index',
        'L3_CH4_CH4_column_volume_mixing_ratio_dry_air',
        'L3_CH4_aerosol_height',
        'L3_CH4_aerosol_optical_depth'
    ]
    # Drop rows where any of the specified columns contain 0-values in X and y
    rows_delete = df_X[columns_with_zeros] != 0
    df_X = df_X[rows_delete.all(axis=1)]
    df_y = df_y[rows_delete.all(axis=1)]
    return df_X, df_y

def drop_ch4_columns(df):
    # Identify columns that contain the word 'CH4' in their headers
    ch4_columns = df.columns[df.columns.str.contains('CH4')]
    # Drop these columns from the DataFrame
    df.drop(columns=ch4_columns, inplace=True)
    return df

def drop_place_id_date(df):
    # Identify column that contains the word 'Place_ID X Date' in headers
    pidd_columns = df.columns[df.columns.str.contains('Place_ID X Date')]
    # Drop column from the DataFrame
    df.drop(columns=pidd_columns, inplace=True)
    return df

def standardize_column_names(df):
    # Transform all column headers: convert to lower case and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Display the first few rows of the DataFrame to check the updated column names
    return df

def ordinal_encoding_dates(df_X, df_y):
    # Sort the DataFrame by the date column
    df_X = df_X.sort_values(by='date')
    df_X['date'] = pd.factorize(df_X['date'])[0]
    df_y = df_y[df_X.index]
    return df_X.reset_index(drop=True), df_y.reset_index(drop=True)

def preprocessing_df(df_X, df_y):
    df_X = drop_columns(df_X)
    df_X = convert_date_and_extract_components(df_X)
    df_X, df_y = drop_zero_rows(df_X, df_y)
    df_X= drop_ch4_columns(df_X)
    df_X= drop_place_id_date(df_X)
    df_X = standardize_column_names(df_X)
    df_X, df_y = ordinal_encoding_dates(df_X, df_y)

    # Print the new shape of the dataframe
    print("New dataframe shape:", df_X.shape, df_y.shape)
    return df_X, df_y


