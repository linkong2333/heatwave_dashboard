import streamlit as st

st.set_page_config(page_title="Urban Heatwave Vulnerability Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor


# ------------------- Streamlit Layout (Refactored) -------------------

def main():
    import json
    import os
    import joblib
    from shapely.geometry import Point
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import seaborn as sns
    import folium
    from streamlit_folium import st_folium
    from branca.colormap import linear

    st.title("Urban Heatwave Vulnerability Dashboard")
    st.markdown("Visualize heatwave risk distribution and social vulnerability across Hong Kong districts.")

    # ---------------------- Data Loading with Caching ----------------------
    @st.cache_data
    def load_training_data():
        data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        return df

    @st.cache_data
    def load_pop_and_boundary():
        # Population
        root_path = "人口"
        districts = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        population_data = []
        for district in districts:
            try:
                path_m01 = os.path.join(root_path, district, "Table M01.csv")
                path_m02 = os.path.join(root_path, district, "Table M02.csv")
                m01 = pd.read_csv(path_m01, skiprows=4)
                m02 = pd.read_csv(path_m02, skiprows=4)
                m01.columns = m01.columns.str.strip()
                m02.columns = m02.columns.str.strip()
                m01 = m01[['年', 'Unnamed: 1', 'Unnamed: 2', '2024']].dropna()
                m02 = m02[['年', 'Unnamed: 1', '2024.3']].dropna()
                m01_total = m01[m01['Unnamed: 1'] == '總計']
                children_pop = m01_total[m01_total['Unnamed: 2'] == '0 - 14歲']['2024'].values[0]
                elderly_pop = m01_total[m01_total['Unnamed: 2'] == '65歲及以上']['2024'].values[0]
                total_pop = m01_total[m01_total['Unnamed: 2'] == '總計']['2024'].values[0]
                children_ratio = children_pop / total_pop
                elderly_ratio = elderly_pop / total_pop
                m02_total = m02[m02['Unnamed: 1'] == '合計']
                income_median = m02_total['2024.3'].values[0]
                population_data.append({
                    'District_CN': district,
                    'Elderly_Ratio': elderly_ratio,
                    'Children_Ratio': children_ratio,
                    'Income_Median': income_median
                })
            except Exception:
                continue
        pop_df = pd.DataFrame(population_data)
        district_mapping = {
            '中西区': 'Central and Western District',
            '东区': 'Eastern District',
            '南区': 'Southern District',
            '湾仔': 'Wan Chai District',
            '离岛': 'Islands District',
            '油尖旺': 'Yau Tsim Mong District',
            '观塘': 'Kwun Tong District',
            '深水涉': 'Sham Shui Po District',
            '九龙城': 'Kowloon City District',
            '黄大仙': 'Wong Tai Sin District',
            '葵青': 'Kwai Tsing District',
            '荃湾': 'Tsuen Wan District',
            '屯门': 'Tuen Mun District',
            '沙田': 'Sha Tin District',
            '西贡': 'Sai Kung District',
            '北区': 'North District',
            '大甫': 'Tai Po District',
        }
        pop_df['District'] = pop_df['District_CN'].map(district_mapping)
        pop_df = pop_df.drop(columns=['District_CN'])
        # District boundaries
        districts_gdf = gpd.read_file("DistrictBoundary_SHP/DCD.shp")
        districts_gdf = districts_gdf.to_crs("EPSG:4326")
        districts_gdf = districts_gdf.rename(columns={'NAME_EN': 'District'})
        districts_gdf = districts_gdf[['District', 'geometry']]
        return pop_df, districts_gdf

    @st.cache_data
    def load_heatwave_points():
        data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        def extract_lon_lat(geo_str):
            try:
                geo_obj = json.loads(geo_str)
                coords = geo_obj.get('coordinates', [None, None])
                return pd.Series({"longitude": coords[0], "latitude": coords[1]})
            except Exception:
                return pd.Series({"longitude": None, "latitude": None})
        if '.geo' in df.columns:
            coords_df = df['.geo'].apply(extract_lon_lat)
            df = pd.concat([df, coords_df], axis=1)
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf

    @st.cache_data
    def get_vulnerability_gdf():
        pop_df, districts_gdf = load_pop_and_boundary()
        gdf = load_heatwave_points()
        gdf_with_district = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")
        gdf_with_district = gdf_with_district.merge(pop_df, on='District', how='left')
        scaler = MinMaxScaler()
        gdf_with_district['LST_norm'] = scaler.fit_transform(gdf_with_district[['LST_Celsius']])
        gdf_with_district['Elderly_norm'] = scaler.fit_transform(gdf_with_district[['Elderly_Ratio']])
        gdf_with_district['Children_norm'] = scaler.fit_transform(gdf_with_district[['Children_Ratio']])
        gdf_with_district['Income_Median_norm'] = 1 - scaler.fit_transform(gdf_with_district[['Income_Median']])
        gdf_with_district['Vulnerability_Index'] = (
            gdf_with_district['LST_norm'] * 0.5 +
            gdf_with_district['Elderly_norm'] * 0.2 +
            gdf_with_district['Children_norm'] * 0.2 +
            gdf_with_district['Income_Median_norm'] * 0.1
        )
        return gdf_with_district, districts_gdf

    # ---------------------- Visualization Sections ----------------------

    st.header("Model Training & Data Preprocessing")
    st.write("This section shows the data preprocessing and model training steps.")
    df = load_training_data()
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    # Extract lon/lat
    if '.geo' in df.columns:
        def extract_lon_lat(geo_str):
            try:
                geo_obj = json.loads(geo_str)
                coords = geo_obj.get('coordinates', [None, None])
                return pd.Series({"longitude": coords[0], "latitude": coords[1]})
            except Exception:
                return pd.Series({"longitude": None, "latitude": None})
        coords_df = df['.geo'].apply(extract_lon_lat)
        df = pd.concat([df, coords_df], axis=1)
    required_cols = ['LST_Celsius', 'Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH', 'longitude', 'latitude']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
    else:
        st.success("All required columns present.")
    df = df.dropna(subset=required_cols)
    Q1 = df['LST_Celsius'].quantile(0.25)
    Q3 = df['LST_Celsius'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['LST_Celsius'] >= lower_bound) & (df['LST_Celsius'] <= upper_bound)]
    st.write(f"After outlier removal, sample count: {len(df)}")
    X = df[['Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH']]
    y = df['LST_Celsius']
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )
    @st.cache_resource
    def train_model(X_train, y_train):
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, y_train)
        return regressor
    rf_regressor = train_model(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Test set R²: {r2:.2f}")
    st.write(f"Test set MAE: {mae:.2f}")
    df_test = df_test.copy()
    df_test['LST_Predicted'] = y_pred
    st.subheader("Test Set Actual vs Predicted LST_Celsius Statistics")
    st.write(df_test[['LST_Celsius', 'LST_Predicted']].describe())
    # Plot actual vs predicted spatial
    geometry = [Point(xy) for xy in zip(df_test['longitude'], df_test['latitude'])]
    gdf = gpd.GeoDataFrame(df_test, geometry=geometry, crs="EPSG:4326")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    gdf.plot(column='LST_Predicted', cmap='coolwarm', legend=True, markersize=50, ax=ax1)
    ax1.set_title("Predicted LST_Celsius (Random Forest Regression)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    gdf.plot(column='LST_Celsius', cmap='coolwarm', legend=True, markersize=50, ax=ax2)
    ax2.set_title("Actual LST_Celsius")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    st.pyplot(fig2)

    st.header("Heatwave Vulnerability Map")
    gdf_with_district, districts_gdf = get_vulnerability_gdf()
    # Create static map with matplotlib
    fig_map, ax_map = plt.subplots(figsize=(10, 8))
    # Plot district boundaries
    districts_gdf.boundary.plot(ax=ax_map, linewidth=1, color='black')
    # Prepare colormap
    norm = colors.Normalize(
        vmin=gdf_with_district['Vulnerability_Index'].min(),
        vmax=gdf_with_district['Vulnerability_Index'].max()
    )
    cmap = cm.get_cmap('RdYlGn_r')
    # Plot points
    gdf_with_district.plot(
        ax=ax_map,
        column='Vulnerability_Index',
        cmap=cmap,
        norm=norm,
        markersize=18,
        alpha=0.7,
        legend=False
    )
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig_map.colorbar(sm, ax=ax_map, fraction=0.025, pad=0.01)
    cbar.set_label('Urban Heatwave Vulnerability Index')
    ax_map.set_title("Urban Heatwave Vulnerability Map (Static)", fontsize=16)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    st.pyplot(fig_map)

    st.header("District-wise Average Vulnerability Index")
    gdf_with_district, _ = get_vulnerability_gdf()
    # Remove empty or invalid values before plotting
    valid_gdf = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
    # Remove empty or invalid district names
    valid_gdf = valid_gdf[valid_gdf['District'].notnull() & (valid_gdf['District'] != "")]
    district_mean = valid_gdf.groupby('District')['Vulnerability_Index'].mean()
    # Remove any districts with NaN mean
    district_mean = district_mean.dropna()
    # Sort the district_mean
    district_mean = district_mean.sort_values(ascending=False)
    # Remove any empty columns
    district_mean = district_mean[district_mean.index != ""]
    norm = colors.Normalize(vmin=district_mean.min(), vmax=district_mean.max())
    cmap = cm.get_cmap('RdYlGn_r')
    bar_colors = [cmap(norm(value)) for value in district_mean]
    fig, ax = plt.subplots(figsize=(14, 8))
    district_mean.plot(kind='bar', color=bar_colors, ax=ax)
    ax.set_title("Average Urban Heatwave Vulnerability Index by District", fontsize=16)
    ax.set_xlabel("District")
    ax.set_ylabel("Average Vulnerability Index")
    # Adjust y-axis range to fit the data tightly
    ymin = max(0, district_mean.min() - 0.01)
    ymax = min(1, district_mean.max() + 0.01)
    ax.set_ylim(ymin, ymax)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    csv = district_mean.reset_index().to_csv(index=False)
    st.download_button(
        label="Download District Vulnerability CSV",
        data=csv,
        file_name='district_vulnerability.csv',
        mime='text/csv',
    )

    st.header("Distribution of Vulnerability Index in Districts")
    gdf_with_district, _ = get_vulnerability_gdf()
    district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
    district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index
    mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
    norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
    cmap = cm.get_cmap('RdYlGn_r')
    palette = {district: cmap(norm(value)) for district, value in mean_values.items()}
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=district_vi,
        x='District',
        y='Vulnerability_Index',
        order=district_order,
        palette=palette,
        showfliers=False
    )
    ax.set_title("Distribution of Urban Heatwave Vulnerability Index by District", fontsize=16)
    ax.set_xlabel("District")
    ax.set_ylabel("Vulnerability Index")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit app entry point
if __name__ == "__main__":
    main()