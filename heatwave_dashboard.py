import streamlit as st

st.set_page_config(page_title="Urban Heatwave Vulnerability Dashboard", layout="wide")

# ------------------- Global Font Size Styling -------------------
# Increase overall font size for readability and presentations
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }
    /* Make titles and headers larger for presentation */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-size: 2.1rem !important;
    }
    .stApp .stCaption, .stApp .caption {
        font-size: 1.1rem !important;
    }
    /* Markdown list text size */
    ul, ol {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    from PIL import Image

    st.title("Urban Heatwave Vulnerability Dashboard")

    # Display images and text in aligned columns
    img1 = Image.open("img/img1.jpg")
    img2 = Image.open("img/img2.jpg")
    img3 = Image.open("img/img3.jpg")  # Flowchart image

    # First row: img1 and its text
    col_img1, col_txt1 = st.columns([1, 2])
    with col_img1:
        st.image(img1, use_container_width=True)
    with col_txt1:
        st.markdown("""
        <span style="font-size:26px; font-weight:bold; color:#D62828;">Research Background</span>
        <ul style="font-size:19px;">
        <li><span style="font-weight:bold; color:#003049;">Hong Kong is experiencing increasingly frequent and intense heatwaves</span> due to climate change and rapid urbanization.</li>
        <li>The <span style="font-weight:bold; color:#003049;">urban heat island effect</span> exacerbates high temperatures in densely populated districts.</li>
        <li>Heatwaves pose significant health risks, especially to <span style="font-weight:bold; color:#003049;">vulnerable groups</span> such as the elderly, children, and low-income residents.</li>
        </ul>
        <span style="font-size:26px; font-weight:bold; color:#D62828;">Objective</span>
        <ul style="font-size:19px;">
        <li>This dashboard aims to visualize the spatial distribution of heatwave risk and social vulnerability across Hong Kong districts.</li>
        <li>It integrates land surface temperature, demographic, and socioeconomic data to identify communities most at risk.</li>
        <li>Supports targeted adaptation and mitigation strategies for urban heatwave resilience.</li>
        </ul>
        """, unsafe_allow_html=True)

    # Second row: img2 and its text
    col_txt2, col_img2 = st.columns([2, 1])
    with col_txt2:
        st.markdown("""
        <span style="font-size:26px; font-weight:bold; color:#D62828;">Heatwave Vulnerability Assessment Overview</span>
        <ul style="font-size:19px;">
        <li>This section provides an overview of the methodology and approach for evaluating heatwave vulnerability in Hong Kong.</li>
        <li>The assessment integrates multiple data sources, including population demographics, land surface temperature, and urban form variables.</li>
        <li>The goal is to identify spatial patterns of vulnerability and inform targeted adaptation strategies.</li>
        </ul>
        """, unsafe_allow_html=True)
    with col_img2:
        st.image(img2, use_container_width=True)

    # ---------------------- Modelling, Data, and Results Section (Combined) ----------------------
    st.header("Modelling, Data Processing, and Results")
    col_data_txt, col_data_img = st.columns([2, 1])
    with col_data_txt:
        st.markdown("""
        <div style="font-size:22px; font-weight:bold; color:#457B9D; margin-bottom: 10px;">Data Sources, Processing &amp; Modelling Workflow</div>
        <ul style="font-size:18px; margin-left: 10px;">
          <li><b style="color:#003049;">Land Surface Temperature (LST):</b> <span style="color:#222;">Satellite remote sensing</span></li>
          <li><b style="color:#003049;">Socioeconomic &amp; Demographic Data:</b> <span style="color:#222;">Age structure, income (Hong Kong Census &amp; Statistics Dept.)</span></li>
          <li><b style="color:#003049;">Urban Form Variables:</b>
            <ul style="margin-left: 15px;">
              <li>Building density</li>
              <li>Green cover</li>
              <li>NDVI</li>
              <li>Road density</li>
            </ul>
          </li>
          <li><b style="color:#003049;">Data Preprocessing:</b>
            <ul style="margin-left: 15px;">
              <li>Outlier removal (IQR method)</li>
              <li>Normalization (Min-Max scaling)</li>
              <li>Spatial joining of point and district data</li>
            </ul>
          </li>
          <li>
            <b style="color:#003049;">Feature Selection &amp; Model Training:</b>
            <ul style="margin-left: 15px;">
              <li>Features: LST, Albedo, Building Density, Green Cover, NDVI, RH, Elderly Ratio, Children Ratio, Median Income</li>
              <li><span style="font-weight:bold; color:#D62828;">Random Forest Regression</span> for LST prediction</li>
              <li>Train-test split (80/20%) and evaluation (R², MAE, MSE)</li>
            </ul>
          </li>
          <li>
            <b style="color:#003049;">Vulnerability Index Construction:</b>
            <ul style="margin-left: 15px;">
              <li>Weighted integration of normalized LST, Elderly Ratio, Children Ratio, and Inverse Median Income</li>
              <li style="margin-top: 3px;">Formula: <span style="color:#D62828; font-weight:bold;">VI = 0.5×LST + 0.2×Elderly + 0.2×Children + 0.1×(1-Income)</span></li>
            </ul>
          </li>
          <li>
            <b style="color:#003049;">Spatial Analysis &amp; Visualization:</b>
            <ul style="margin-left: 15px;">
              <li>District-level aggregation and comparison</li>
              <li>Spatial mapping of predicted LST and vulnerability index</li>
              <li>Boxplots and bar charts for district-wise distribution</li>
            </ul>
          </li>
        </ul>
        <div style="margin-top: 10px; font-size:18px;">
          <b style="color:#D62828;">&#9656; This integrated workflow provides a robust, data-driven approach for urban heatwave vulnerability assessment.</b>
        </div>
        """, unsafe_allow_html=True)
    with col_data_img:
        st.image(img3, caption="Data Integration and Modelling Workflow", use_container_width=True)

    st.markdown("""
    ---
    <div style="font-size:22px; font-weight:bold; color:#457B9D; margin-bottom: 10px;">
        Data Preprocessing, Model Training, and Key Results
    </div>
    <ul style="font-size:18px; margin-left: 10px;">
      <li>
        <b style="color:#003049;">Data Preprocessing:</b>
        Outliers in LST were removed using the IQR method, and all variables were normalized to ensure comparability. Spatial joining linked point-level measurements with district-level demographic and socioeconomic data.
      </li>
      <li>
        <b style="color:#003049;">Model Training:</b>
        A Random Forest regression model was trained to predict LST from urban form and environmental variables. The model achieved strong predictive performance:
        <ul style="margin-left: 15px;">
          <li>Test set R² and MAE (see below for values and distribution plots)</li>
        </ul>
      </li>
      <li>
        <b style="color:#003049;">Vulnerability Index:</b>
        The vulnerability index (VI) integrates predicted LST, elderly and children population ratios, and inverse median income, highlighting communities most at risk.
      </li>
      <li>
        <b style="color:#003049;">Results Visualization:</b>
        <ul style="margin-left: 15px;">
          <li>Spatial maps show predicted LST and vulnerability index distributions across Hong Kong.</li>
          <li>Bar charts and boxplots summarize district-wise vulnerability, supporting targeted adaptation strategies.</li>
        </ul>
      </li>
    </ul>
    """, unsafe_allow_html=True)

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

    st.subheader("Model Training & Data Preprocessing")
    st.write("Below are the steps and results from model training and data preparation:")
    df = load_training_data()
    st.markdown("<span style='font-size:20px; font-weight:bold;'>Raw Data Preview:</span>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:16px;'>This table shows a sample of the raw data used for model training and analysis. "
        "Columns include land surface temperature (LST), urban form variables, and geospatial coordinates. "
        "The data is preprocessed before use in the vulnerability assessment workflow.</span>",
        unsafe_allow_html=True,
    )
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
    st.write(f"<span style='font-size:19px;'><b>Test set R²:</b> {r2:.2f}</span>", unsafe_allow_html=True)
    st.write(f"<span style='font-size:19px;'><b>Test set MAE:</b> {mae:.2f}</span>", unsafe_allow_html=True)
    df_test = df_test.copy()
    df_test['LST_Predicted'] = y_pred
    st.markdown("<span style='font-size:20px; font-weight:bold;'>Test Set Actual vs Predicted LST_Celsius Statistics:</span>", unsafe_allow_html=True)
    st.write(df_test[['LST_Celsius', 'LST_Predicted']].describe())
    # Plot actual vs predicted spatial
    st.markdown("<span style='font-size:20px; font-weight:bold;'>Spatial Distribution of Predicted and Actual LST:</span>", unsafe_allow_html=True)
    geometry = [Point(xy) for xy in zip(df_test['longitude'], df_test['latitude'])]
    gdf = gpd.GeoDataFrame(df_test, geometry=geometry, crs="EPSG:4326")
    # Arrange the two LST maps side by side and adjust their size for better layout
    col_pred, col_actual = st.columns(2)
    # Adjust LST_figsize: width +20%, height -20%, then shrink by 15%
    orig_LST_figsize = (5.5, 4.5)
    LST_figsize = (orig_LST_figsize[0] * 1.2, orig_LST_figsize[1] * 0.8)
    LST_figsize = (LST_figsize[0] * 0.85, LST_figsize[1] * 0.85)
    # Font sizes: shrink by 30% then by 15% (total shrink = 0.7 * 0.85 = 0.595)
    def shrink(fs): return int(round(fs * 0.595))
    with col_pred:
        fig1, ax1 = plt.subplots(figsize=LST_figsize)
        gdf.plot(column='LST_Predicted', cmap='coolwarm', legend=True, markersize=22, ax=ax1,
                 legend_kwds={'label': "Predicted LST (°C)", 'shrink': 0.8, 'orientation': 'vertical'})
        # Title: break line at parentheses to avoid overlap with legend
        ax1.set_title("Predicted LST_Celsius", fontsize=shrink(16))
        ax1.set_xlabel("Longitude", fontsize=shrink(13))
        ax1.set_ylabel("Latitude", fontsize=shrink(13))
        ax1.tick_params(axis='both', labelsize=shrink(11))
        # Make legend font larger (but shrink by 30%)
        leg = ax1.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(shrink(13))
            leg.set_title(leg.get_title().get_text(), prop={'size': shrink(14)})
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("<span style='font-size:17px;'>Predicted Land Surface Temperature (LST) values across Hong Kong, generated by the Random Forest regression model using urban form and environmental predictors.</span>", unsafe_allow_html=True)
    with col_actual:
        fig2, ax2 = plt.subplots(figsize=LST_figsize)
        gdf.plot(column='LST_Celsius', cmap='coolwarm', legend=True, markersize=22, ax=ax2,
                 legend_kwds={'label': "Actual LST (°C)", 'shrink': 0.8, 'orientation': 'vertical'})
        ax2.set_title("Actual LST_Celsius", fontsize=shrink(16))
        ax2.set_xlabel("Longitude", fontsize=shrink(13))
        ax2.set_ylabel("Latitude", fontsize=shrink(13))
        ax2.tick_params(axis='both', labelsize=shrink(11))
        leg = ax2.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(shrink(13))
            leg.set_title(leg.get_title().get_text(), prop={'size': shrink(14)})
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("<span style='font-size:17px;'>Actual observed LST values from satellite remote sensing, showing the measured spatial pattern of surface temperature.</span>", unsafe_allow_html=True)

    st.subheader("Heatwave Vulnerability Index: Spatial and District Results")
    gdf_with_district, districts_gdf = get_vulnerability_gdf()
    # Vulnerability map: adjust figure size and shrink all text by 30% then by 15%
    # Original size: (8, 6.4) -> width *1.2, height *0.8, then *0.85
    orig_map_figsize = (8, 6.4)
    map_figsize = (orig_map_figsize[0]*1.2, orig_map_figsize[1]*0.8)
    map_figsize = (map_figsize[0]*0.85, map_figsize[1]*0.85)
    fig_map, ax_map = plt.subplots(figsize=map_figsize)
    districts_gdf.boundary.plot(ax=ax_map, linewidth=1, color='black')
    norm = colors.Normalize(
        vmin=gdf_with_district['Vulnerability_Index'].min(),
        vmax=gdf_with_district['Vulnerability_Index'].max()
    )
    cmap = cm.get_cmap('RdYlGn_r')
    gdf_with_district.plot(
        ax=ax_map,
        column='Vulnerability_Index',
        cmap=cmap,
        norm=norm,
        markersize=14,
        alpha=0.7,
        legend=False
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig_map.colorbar(sm, ax=ax_map, fraction=0.025, pad=0.01)
    # Shrink font sizes by 30% then by 15%
    def shrink(fs): return int(round(fs * 0.595))
    cbar.set_label('Urban Heatwave Vulnerability Index', fontsize=shrink(16))
    cbar.ax.tick_params(labelsize=shrink(15))
    ax_map.set_title("Urban Heatwave Vulnerability Map (Static)", fontsize=shrink(22))
    ax_map.set_xlabel("Longitude", fontsize=shrink(17))
    ax_map.set_ylabel("Latitude", fontsize=shrink(17))
    ax_map.tick_params(axis='both', labelsize=shrink(14))
    ax_map.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    st.pyplot(fig_map)
    st.caption("<span style='font-size:17px;'>This map visualizes the spatial distribution of the Urban Heatwave Vulnerability Index, highlighting areas where demographic, socioeconomic, and thermal risk factors combine to increase vulnerability.</span>", unsafe_allow_html=True)

    # Arrange the bar chart and boxplot side by side, reduce their size, and add captions
    col_bar, col_box = st.columns(2)
    # Shrink function for font sizes (30% then 15%)
    def shrink(fs): return int(round(fs * 0.595))
    with col_bar:
        st.markdown("<span style='font-size:20px; font-weight:bold;'>District-wise Average Vulnerability Index:</span>", unsafe_allow_html=True)
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
        # Adjust figure size: width unchanged, height increased by 50%
        orig_bar_figsize = (5.3, 3.7)
        bar_figsize = (orig_bar_figsize[0], orig_bar_figsize[1]*1.2)
        fig, ax = plt.subplots(figsize=bar_figsize)
        district_mean.plot(kind='bar', color=bar_colors, ax=ax)
        ax.set_title("Average Vulnerability Index by District", fontsize=shrink(15))
        ax.set_xlabel("District", fontsize=shrink(12))
        ax.set_ylabel("Average Vulnerability Index", fontsize=shrink(12))
        # Reduce x-axis (district) label font size
        ax.tick_params(axis='x', labelsize=shrink(9))
        ax.tick_params(axis='y', labelsize=shrink(11))
        ymin = max(0, district_mean.min() - 0.01)
        ymax = min(1, district_mean.max() + 0.01)
        ax.set_ylim(ymin, ymax)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("<span style='font-size:17px;'>Bar chart showing the mean heatwave vulnerability index for each district. Higher values indicate greater vulnerability due to combined thermal and social risk factors.</span>", unsafe_allow_html=True)
        csv = district_mean.reset_index().to_csv(index=False)
        st.download_button(
            label="Download District Vulnerability CSV",
            data=csv,
            file_name='district_vulnerability.csv',
            mime='text/csv',
        )

    with col_box:
        st.markdown("<span style='font-size:20px; font-weight:bold;'>Distribution of Vulnerability Index in Districts:</span>", unsafe_allow_html=True)
        gdf_with_district, _ = get_vulnerability_gdf()
        district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
        district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index
        mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
        norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
        cmap = cm.get_cmap('RdYlGn_r')
        palette = {district: cmap(norm(value)) for district, value in mean_values.items()}
        # Adjust figure size: width unchanged, height increased by 50%
        orig_box_figsize = (5.3, 3.7)
        box_figsize = (orig_box_figsize[0], orig_box_figsize[1]*1.2)
        fig, ax = plt.subplots(figsize=box_figsize)
        sns.boxplot(
            data=district_vi,
            x='District',
            y='Vulnerability_Index',
            order=district_order,
            palette=palette,
            showfliers=False
        )
        ax.set_title("Vulnerability Index Distribution by District", fontsize=shrink(15))
        ax.set_xlabel("District", fontsize=shrink(12))
        ax.set_ylabel("Vulnerability Index", fontsize=shrink(12))
        # Reduce x-axis (district) label font size
        ax.tick_params(axis='x', labelsize=shrink(9))
        ax.tick_params(axis='y', labelsize=shrink(11))
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("<span style='font-size:17px;'>Boxplot visualizing the spread and variability of vulnerability index scores within each district, identifying both high-risk and lower-risk areas.</span>", unsafe_allow_html=True)


# Streamlit app entry point
if __name__ == "__main__":
    main()