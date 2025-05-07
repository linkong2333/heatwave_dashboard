import streamlit as st
import os
import seaborn as sns
import tempfile
import zipfile

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
    <script>
        // 自动滚动到顶部的函数
        const scrollToTop = () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
        // 监听按钮点击
        window.addEventListener('load', function() {
            const observer = new MutationObserver((mutations) => {
                scrollToTop();
            });
            observer.observe(document.querySelector('.main'), {
                childList: true,
                subtree: true
            });
        });
    </script>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.markdown("""
<style>
.stButton > button {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Streamlit Layout (Refactored) -------------------

def main():
    import os
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.preprocessing import MinMaxScaler
    from PIL import Image
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    def navigate_to_page(page_num):
        st.session_state['page_number'] = page_num
        st.rerun()

    def add_top_nav_button():
        col1, col2, col3 = st.columns([2, 0.7, 2])
        with col2:
            if st.session_state['page_number'] > 1:
                if st.button("⬆️", help="返回上一页", key="top_nav"):
                    navigate_to_page(st.session_state['page_number'] - 1)

    def add_bottom_nav_button():
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 0.7, 2])
        with col2:
            if st.session_state['page_number'] < 4:
                if st.button("⬇️", help="前往下一页", key="bottom_nav"):
                    navigate_to_page(st.session_state['page_number'] + 1)

    @st.cache_data
    def load_training_data(point_path="Heatwave_Training_Data_With_RoadDensity (4).csv"):
        df = pd.read_csv(point_path)
        df.columns = df.columns.str.strip()
        return df

    @st.cache_data
    def load_pop_and_boundary(pop_path="人口"):
        if os.path.isfile(pop_path) and pop_path.endswith('.csv'):
            pop_df = pd.read_csv(pop_path)
        else:
            root_path = pop_path
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
        districts_gdf = gpd.read_file("DistrictBoundary_SHP/DCD.shp")
        districts_gdf = districts_gdf.to_crs("EPSG:4326")
        districts_gdf = districts_gdf.rename(columns={'NAME_EN': 'District'})
        districts_gdf = districts_gdf[['District', 'geometry']]
        return pop_df, districts_gdf

    @st.cache_data
    def load_heatwave_points(point_path="Heatwave_Training_Data_With_RoadDensity (4).csv"):
        import json
        df = pd.read_csv(point_path)
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
    def get_vulnerability_gdf(w1, w2, w3, w4, pop_path="人口", point_path="Heatwave_Training_Data_With_RoadDensity (4).csv"):
        pop_df, districts_gdf = load_pop_and_boundary(pop_path)
        gdf = load_heatwave_points(point_path)
        gdf_with_district = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")
        gdf_with_district = gdf_with_district.merge(pop_df, on='District', how='left')
        scaler = MinMaxScaler()
        gdf_with_district['LST_norm'] = scaler.fit_transform(gdf_with_district[['LST_Celsius']])
        gdf_with_district['Elderly_norm'] = scaler.fit_transform(gdf_with_district[['Elderly_Ratio']])
        gdf_with_district['Children_norm'] = scaler.fit_transform(gdf_with_district[['Children_Ratio']])
        gdf_with_district['Income_Median_norm'] = 1 - scaler.fit_transform(gdf_with_district[['Income_Median']])
        gdf_with_district['Vulnerability_Index'] = (
            gdf_with_district['LST_norm'] * w1 +
            gdf_with_district['Elderly_norm'] * w2 +
            gdf_with_district['Children_norm'] * w3 +
            gdf_with_district['Income_Median_norm'] * w4
        )
        return gdf_with_district, districts_gdf

    def get_img(fname):
        try:
            return Image.open(fname)
        except Exception:
            return None

    def page1():
        add_top_nav_button()
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center; margin-top:70px; margin-bottom:40px;">
                <span style="font-size:2.7rem; font-weight:bold; color:#003049; text-align:center;">
                Urban Heatwave Susceptibility Analysis and Visualization System for Hong Kong
                </span>
                <span style="margin-top:40px; font-size:1.5rem; font-weight:bold; color:#457B9D;">
                Liu Bingyi, Li Chenhan, Liang Zixin, Huang Baihui
                </span>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="text-align:center; margin-top:30px;">
                <span style="font-size:1.1rem; color:#222;">A comprehensive dashboard for assessing and visualizing urban heatwave vulnerability in Hong Kong, integrating climate, demographic, and socioeconomic data.</span>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("---")
        st.info("Use the sidebar to navigate to different sections of the dashboard.")
        img1 = get_img("img/img1.jpg")
        img2 = get_img("img/img2.jpg")
        col_img1, col_txt1 = st.columns([1, 2])
        with col_img1:
            if img1:
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
            <li>This dashboard aims to <b style="color:#003049;">visualize the spatial distribution of heatwave risk</b> and <b style="color:#003049;">social vulnerability</b> across Hong Kong districts.</li>
            <li>It <b style="color:#003049;">integrates land surface temperature, demographic, and socioeconomic data</b> to identify communities most at risk.</li>
            <li>Supports <b style="color:#003049;">targeted adaptation and mitigation strategies</b> for urban heatwave resilience.</li>
            </ul>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <style>
        .overview-list {
            list-style-type: disc !important;
            padding-left: 1.2em !important;
            margin-top: 1em !important;
        }
        .overview-list li {
            padding-left: 0.5em !important;
            margin-bottom: 1em !important;
            display: list-item !important;
            color: black !important;
        }
        .overview-list li::marker {
            color: black !important;
        }
        </style>
        <div>
        <span style="font-size:26px; font-weight:bold; color:#D62828;">Heatwave Vulnerability Assessment Overview</span>
        <ul class="overview-list" style="font-size:19px;">
        <li>This section provides an overview of the <b style="color:#003049;">methodology and approach</b> for evaluating heatwave vulnerability in Hong Kong.</li>
        <li>The assessment <b style="color:#003049;">integrates multiple data sources</b>, including <b style="color:#003049;">population demographics, land surface temperature, and urban form variables</b>.</li>
        <li>The goal is to identify <b style="color:#003049;">spatial patterns of vulnerability</b> and inform <b style="color:#003049;">targeted adaptation strategies</b>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        img3 = get_img("img/img3.jpg")
        if img3:
            col1, col2, col3 = st.columns([1,7,1])
            with col2:
                st.image(img3, width=int(img3.width * 0.96))
                st.markdown("""
                <div style="text-align: center; color: #666666; font-size: 0.9em; margin-top: -10px;">
                Technical Roadmap: Data Processing and Model Development Pipeline
                </div>
                """, unsafe_allow_html=True)
        add_bottom_nav_button()
        return

    def page2():
        add_top_nav_button()
        st.markdown("<h2 style='color:#003049;'>Model</h2>", unsafe_allow_html=True)
        st.markdown("""
        We generated 50,000 random sampling points across the Hong Kong region and extracted environmental and urban features at each location to serve as model input. The data preprocessing process included outlier removal (via the IQR method), min-max normalization, and spatial joining to integrate geographic and socioeconomic attributes.
        """)
        st.markdown("### Features Used")
        feature_table = pd.DataFrame({
            'Feature': ['LST', 'NDVI', 'Albedo', 'Building Density (CSDI)', 'Relative Humidity (RH)', 'Green Cover'],
            'Description': [
                'Derived from Landsat 8 thermal band (ST_B10) and calibrated to Celsius',
                'Calculated using Landsat 8 near-infrared band (SR_B5) and red band (SR_B4)',
                'Estimated through a weighted combination of visible and shortwave infrared bands',
                'Density of buildings in urban areas',
                'From the National Earth System Science Data Center',
                'Derived by thresholding NDVI'
            ],
            'Scientific Rationale': [
                'Obtained from satellite data to assess heat risk',
                'Reflects vegetation cover, which cools the land via evapotranspiration, negatively correlated with surface temperature, mitigates urban heat island effects',
                'Lower albedo leads to greater heat absorption, urban areas typically have lower albedo, contributing to heat accumulation',
                'High building density reduces ventilation, increases heat retention, intensifies local heat island effects',
                'Affects thermal comfort and heat stress risk, lower RH increases health risks under high temperatures, essential for assessing thermal vulnerability',
                'Reflects cooling capacity in urban environments, areas with more greenery offer better thermal protection for residents'
            ]
        })
        st.table(feature_table)
        st.markdown("<span style='font-size:20px; font-weight:bold;'>Raw Data Preview:</span>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:16px;'>Below shows a sample of our preprocessed dataset, containing the key features used for model training. Each row represents a sampling point with its corresponding environmental and urban form characteristics.</span>", unsafe_allow_html=True)
        df = load_training_data()
        st.dataframe(df.head())
        st.markdown("""
        Following preprocessing, we randomly split the dataset into a training set (80%) and a test set (20%). A Random Forest Regression model was trained on the former and evaluated on the latter to assess its predictive capability.
        """)
        st.markdown("### Model Performance")
        st.markdown("The model achieved the following evaluation results:")
        X = df[['Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH']]
        y = df['LST_Celsius']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)
        y_pred = rf_regressor.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"<span style='font-size:19px;'><b>R²:</b> {r2:.2f} – strong explanatory power</span>", unsafe_allow_html=True)
        st.write(f"<span style='font-size:19px;'><b>Mean Absolute Error:</b> {mae:.2f}°C – low average prediction error</span>", unsafe_allow_html=True)
        st.markdown("### Spatial Distribution of Predicted and Actual LST:")
        if '.geo' in df.columns:
            import json
            def extract_lon_lat(geo_str):
                try:
                    geo_obj = json.loads(geo_str)
                    coords = geo_obj.get('coordinates', [None, None])
                    return pd.Series({"longitude": coords[0], "latitude": coords[1]})
                except Exception:
                    return pd.Series({"longitude": None, "latitude": None})
            coords_df = df['.geo'].apply(extract_lon_lat)
            df = pd.concat([df, coords_df], axis=1)
        X = df[['Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH']]
        y = df['LST_Celsius']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_indices = y_test.index
        df_test = pd.DataFrame({
            'LST_Celsius': y_test,
            'LST_Predicted': y_pred,
            'longitude': df.iloc[test_indices]['longitude'].values,
            'latitude': df.iloc[test_indices]['latitude'].values
        })
        geometry = [Point(xy) for xy in zip(df_test['longitude'], df_test['latitude'])]
        gdf = gpd.GeoDataFrame(df_test, geometry=geometry, crs="EPSG:4326")
        col_pred, col_actual = st.columns(2)
        orig_LST_figsize = (5.5, 4.5)
        LST_figsize = (orig_LST_figsize[0]*1.2, orig_LST_figsize[1]*0.8)
        LST_figsize = (LST_figsize[0]*0.85, LST_figsize[1]*0.85)
        def shrink(fs): return int(round(fs * 0.595))
        from matplotlib.colors import LinearSegmentedColormap
        colors_lst = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
                     '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        cmap_lst = LinearSegmentedColormap.from_list('custom', colors_lst)
        vmin, vmax = 25, 45
        with col_pred:
            fig1, ax1 = plt.subplots(figsize=LST_figsize)
            gdf.plot(column='LST_Predicted', cmap=cmap_lst, legend=True, markersize=22, ax=ax1,
                     vmin=vmin, vmax=vmax,
                     legend_kwds={'label': "Predicted LST (°C)", 'shrink': 0.8, 'orientation': 'vertical', 'extend': 'both', 'ticks': range(25, 46, 5)})
            ax1.set_title("Predicted LST_Celsius", fontsize=shrink(16))
            ax1.set_xlabel("Longitude", fontsize=shrink(13))
            ax1.set_ylabel("Latitude", fontsize=shrink(13))
            ax1.tick_params(axis='both', labelsize=shrink(11))
            leg = ax1.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontsize(shrink(13))
                leg.set_title(leg.get_title().get_text(), prop={'size': shrink(14)})
            plt.tight_layout()
            st.pyplot(fig1)
        with col_actual:
            fig2, ax2 = plt.subplots(figsize=LST_figsize)
            gdf.plot(column='LST_Celsius', cmap=cmap_lst, legend=True, markersize=22, ax=ax2,
                     vmin=vmin, vmax=vmax,
                     legend_kwds={'label': "Actual LST (°C)", 'shrink': 0.8, 'orientation': 'vertical', 'extend': 'both', 'ticks': range(25, 46, 5)})
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
        st.caption("<span style='font-size:17px;'>The predicted temperature patterns closely resemble the observed distribution, especially in areas with high building density and low vegetation, suggesting that urban form variables contribute significantly to heat variation.</span>", unsafe_allow_html=True)
        st.markdown("### Insights from Feature Importance")
        st.markdown("We analyzed which features most influenced LST predictions. The feature importance chart provides insights into the drivers of urban heat.")
        importances = rf_regressor.feature_importances_
        feature_names = ['Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH']
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        colors_bar = plt.cm.RdYlGn_r(np.linspace(0, 1, len(feature_importance_df)))
        orig_figsize = (8, 6)
        new_figsize = (orig_figsize[0], orig_figsize[1] * 0.8)
        fig_imp = plt.figure(figsize=new_figsize)
        plt.subplots_adjust(top=0.95, bottom=0.15)
        bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors_bar, height=0.6)
        orig_fontsize = 10
        new_fontsize = orig_fontsize * 0.8
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', va='center', ha='left', fontsize=new_fontsize)
        plt.xlabel('Feature Importance', fontsize=new_fontsize)
        plt.title('Feature Importance from Random Forest Model', fontsize=new_fontsize * 1.2)
        plt.xticks(fontsize=new_fontsize)
        plt.yticks(fontsize=new_fontsize)
        plt.xlim(0, max(importances) + 0.05)
        plt.tight_layout()
        st.pyplot(fig_imp)
        st.markdown("""
        <div style="font-size:20px; line-height:1.8; padding: 30px;">
        <p style="margin-bottom:20px;">As shown in the chart, the model revealed the following insights:</p>
        <p style="margin-bottom:15px;">1. <b style="color:#003049;">Building Density</b> was the most influential factor in predicting surface temperature, suggesting that dense urban structures significantly contribute to urban heat.</p>
        <p style="margin-bottom:15px;">2. <b style="color:#003049;">Albedo</b> also played a major role. Surfaces with low reflectivity (dark roofs, asphalt) tend to absorb more heat, increasing LST.</p>
        <p style="margin-bottom:15px;">3. <b style="color:#003049;">Relative Humidity (RH)</b> ranked third in importance, indicating that atmospheric moisture does influence thermal patterns, but to a moderate degree.</p>
        <p style="margin-bottom:15px;">4. <b style="color:#003049;">Vegetation-related variables</b> showed a mixed impact:
        <ul style="margin-left:25px; margin-top:10px;">
            <li><b style="color:#003049;">NDVI</b> had a noticeable cooling effect (0.18), reflecting the role of vegetation health.</li>
            <li><b style="color:#003049;">Green Cover</b>, however, had minimal contribution (0.001), possibly due to overlap with NDVI or spatial resolution limitations.</li>
        </ul>
        </p>
        <p style="margin-bottom:15px;">In conclusion, our model effectively captures the influence of urban and environmental variables on temperature variation. We also found that <b style="color:#003049;">urban form and surface materials</b> (rather than vegetation alone) are primary drivers of heat distribution in our study area. This directly informs how we construct the Heatwave Vulnerability Index in the next stage of analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        add_bottom_nav_button()
        return

    def page3():
        add_top_nav_button()
        st.markdown("<h2 style='color:#003049;'>Vulnerability Index</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:1.1rem;'>
        <b>Urban Heatwave Vulnerability Index Construction and Spatial Analysis</b><br>
        To systematically assess the urban population's vulnerability to extreme heat events, we constructed a <b>Heatwave Vulnerability Index (VI)</b> using a weighted combination of multiple environmental and socioeconomic indicators.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### • Indicator Design and Rationale")
        st.markdown("""
        Following IPCC's conceptual framework, vulnerability is determined by three components:
        1. → <b>Exposure</b> – measured by predicted land surface temperature (LST), which reflects ambient heat risk.<br>
        2. → <b>Sensitivity</b> – represented by the proportions of elderly (65+) and children (0–14), who are more physiologically vulnerable to heat.<br>
        3. → <b>Adaptive Capacity</b> – inversely represented by median household income, assuming that lower income groups are less equipped to cope.
        """, unsafe_allow_html=True)
        st.markdown("The final VI is calculated as:")
        st.markdown("""
        <div style="text-align: center; font-size: 1.1em; margin: 20px 0; font-style: italic;">
        VI = w1 * LST + w2 * Elderly + w3 * Children + w4 * Income
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Weights (wi) are user-defined (default: 0.5, 0.2, 0.2, 0.1) and sum to 1.0.")
        st.markdown("### Heatwave Vulnerability Index: Spatial and District Results")
        st.markdown("This map shows the pointwise distribution of the VI across the study area. High VI values concentrate in districts with:")
        st.markdown("""
        • <b>Areas of higher vulnerability are concentrated in:</b> high surface temperatures, high elderly populations, and lower income segments. For example, Tuen Mun, Sham Shui Po, Kwun Tong and Kowloon City districts are high-risk hotspots;<br>
        • <b>Areas with more green space and lower LST</b> (e.g. Sai Kung, Wan Chai, and Central and Western Districts) show lower vulnerability.
        """, unsafe_allow_html=True)
        try:
            gdf_with_district, districts_gdf = get_vulnerability_gdf(w1, w2, w3, w4)
            # 地图部分（大尺寸，标题和legend字体小巧）
            fig_map, ax_map = plt.subplots(figsize=(4, 3))
            districts_gdf.boundary.plot(ax=ax_map, linewidth=1, color='black')
            norm = colors.Normalize(vmin=0.35, vmax=0.75)
            cmap = cm.get_cmap('RdYlGn_r')
            gdf_with_district.plot(
                ax=ax_map,
                column='Vulnerability_Index',
                cmap=cmap,
                norm=norm,
                markersize=14,
                alpha=0.7,
                legend=True,
                legend_kwds={
                    'label': 'Urban Heatwave Vulnerability Index',
                    'orientation': 'vertical',
                    'shrink': 0.6,
                    'extend': 'both',
                    'ticks': np.arange(0.3, 0.9, 0.1)
                }
            )
            ax_map.set_title("Urban Heatwave Vulnerability Map (Static)", fontsize=10)
            ax_map.set_xlabel("Longitude", fontsize=8)
            ax_map.set_ylabel("Latitude", fontsize=8)
            ax_map.tick_params(axis='both', labelsize=8)
            # legend字体缩小
            leg = ax_map.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontsize(8)
                leg.set_title(leg.get_title().get_text(), prop={'size': 9})
            plt.tight_layout()
            st.pyplot(fig_map)
            st.caption("<span style='font-size:15px;'>This map visualizes the spatial distribution of the Urban Heatwave Vulnerability Index, highlighting areas where demographic, socioeconomic, and thermal risk factors combine to increase vulnerability.</span>", unsafe_allow_html=True)
            # 下方两个分区图并列一行，无大标题副标题，直接用图内标题
            col_bar, col_box = st.columns(2)
            with col_bar:
                valid_gdf = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
                valid_gdf = valid_gdf[valid_gdf['District'].notnull() & (valid_gdf['District'] != "")]
                district_mean = valid_gdf.groupby('District')['Vulnerability_Index'].mean()
                district_mean = district_mean.dropna()
                district_mean = district_mean.sort_values(ascending=False)
                district_mean = district_mean[district_mean.index != ""]
                norm = colors.Normalize(vmin=district_mean.min(), vmax=district_mean.max())
                cmap = cm.get_cmap('RdYlGn_r')
                bar_colors = [cmap(norm(value)) for value in district_mean]
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                district_mean.plot(kind='bar', color=bar_colors, ax=ax)
                ax.set_title("Average Urban Heatwave Vulnerability Index by District", fontsize=12)
                ax.set_xlabel("District", fontsize=10)
                ax.set_ylabel("Average Vulnerability Index", fontsize=10)
                ax.tick_params(axis='x', labelsize=7, rotation=45)
                ax.tick_params(axis='y', labelsize=10)
                ax.set_xticks(range(len(district_mean)))
                ax.set_xticklabels(district_mean.index, fontsize=7, rotation=45, ha='right')
                ymin = max(0, district_mean.min() - 0.01)
                ymax = min(1, district_mean.max() + 0.01)
                ax.set_ylim(ymin, ymax)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("""
                <span style='font-size:14px;'>
                • Tuen Mun, Sham Shui Po, Kwun Tong and Kowloon City districts are characterized by significantly higher-than-average combined vulnerability. These districts tend to be characterized by: high population density, high proportion of elderly people, and poor green space coverage;<br>
                • Comparatively speaking, the Wan Chai, Central and Western, and Sai Kung districts have lower vulnerability, and their living environment is more conducive to coping with heat waves.
                </span>
                """, unsafe_allow_html=True)
            with col_box:
                district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
                district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index
                mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
                norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
                cmap = cm.get_cmap('RdYlGn_r')
                palette = {district: cmap(norm(value)) for district, value in mean_values.items()}
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                sns.boxplot(
                    data=district_vi,
                    x='District',
                    y='Vulnerability_Index',
                    order=district_order,
                    palette=palette,
                    showfliers=False
                )
                ax.set_title("Distribution of Urban Heatwave Vulnerability Index by District", fontsize=12)
                ax.set_xlabel("District", fontsize=10)
                ax.set_ylabel("Vulnerability Index", fontsize=10)
                ax.tick_params(axis='x', labelsize=7, rotation=45)
                ax.tick_params(axis='y', labelsize=10)
                ax.set_xticks(range(len(district_order)))
                ax.set_xticklabels(district_order, fontsize=7, rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("""
                <span style='font-size:14px;'>
                • Districts such as Wong Tai Sin, Sham Shui Po and Kwai Tsing have a large span of box lines and significant internal imbalances;<br>
                • Some districts (e.g. Sai Kung and Island) have a relatively concentrated distribution, indicating a more balanced internal thermal risk.
                </span>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading vulnerability index map: {e}")
        add_bottom_nav_button()
        return

    def page4():
        add_top_nav_button()
        st.markdown("<h2 style='color:#003049;'>Conclusion</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:20px; line-height:1.8; padding: 30px;">
        <p style="margin-bottom:20px;">
        This heatwave vulnerability assessment reveals several key findings:
        </p>

        <p style="margin-bottom:15px;">
        1. <b style="color:#D62828;">Distribution Pattern:</b><br>
        Heat risk in cities is <b style="color:#003049;">not uniformly distributed</b>, but is influenced by a combination of:
        <ul style="margin-left:25px; margin-top:10px;">
            <li><b style="color:#003049;">Environmental conditions</b></li>
            <li><b style="color:#003049;">Demographic factors</b></li>
            <li><b style="color:#003049;">Socio-economic conditions</b></li>
        </ul>
        </p>

        <p style="margin-bottom:15px;">
        2. <b style="color:#D62828;">Strategic Implications:</b><br>
        Identifying <b style="color:#003049;">high-risk areas</b>, such as <b style="color:#003049;">Tuen Mun, Sham Shui Po, Kwun Tong, and Kowloon City</b>, where both <b style="color:#003049;">heat exposure and population sensitivity</b> are high, allows for more <b style="color:#003049;">targeted adaptation strategies</b> and resource allocation.
        </p>

        <p style="margin-bottom:15px;">
        3. <b style="color:#D62828;">Policy Focus:</b><br>
        Urban policymakers should prioritize <b style="color:#003049;">enhancing the stress-bearing capacity of vulnerable communities</b>. Recommended actions include:
        <ul style="margin-left:25px; margin-top:10px;">
            <li><b style="color:#003049;">Expanding urban green infrastructure</b> and cooling measures</li>
            <li>Delivering <b style="color:#003049;">targeted support for heat-sensitive groups</b> (e.g., elderly and low-income households)</li>
            <li><b style="color:#003049;">Integrating vulnerability assessments</b> into urban planning and disaster preparedness frameworks</li>
        </ul>
        </p>
        </div>
        """, unsafe_allow_html=True)
        add_bottom_nav_button()
        return

    # ----------- Sidebar: Data Upload & Weights (English, compact) -----------
    st.sidebar.markdown("## Data Source", unsafe_allow_html=True)
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "Use example data"
    data_source = st.sidebar.radio(
        "Choose data source:",
        ("Use example data", "Upload your own data"),
        index=0 if st.session_state['data_source'] == "Use example data" else 1,
        key="data_source_radio"
    )
    # 切换数据源时清空上传缓存
    if data_source != st.session_state['data_source']:
        st.session_state['data_source'] = data_source
        st.session_state.pop('uploaded_pop', None)
        st.session_state.pop('uploaded_point', None)
        st.session_state.pop('custom_pop_path', None)
        st.session_state.pop('custom_point_path', None)
        st.rerun()

    default_pop_path = "人口"
    default_point_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
    uploaded_pop = None
    uploaded_point = None
    custom_pop_path = None
    custom_point_path = None
    if data_source == "Upload your own data":
        st.sidebar.markdown("<div style='margin-bottom:8px;'>Upload population data (zip/csv):</div>", unsafe_allow_html=True)
        uploaded_pop = st.sidebar.file_uploader("", type=["zip", "csv"], label_visibility="collapsed", key="uploaded_pop")
        st.sidebar.markdown("<div style='margin-bottom:8px;'>Upload monitoring points (csv):</div>", unsafe_allow_html=True)
        uploaded_point = st.sidebar.file_uploader("", type="csv", label_visibility="collapsed", key="uploaded_point")
        st.sidebar.info("Please ensure your data format matches the example data.")
        if uploaded_pop is not None:
            if uploaded_pop.name.endswith('.zip'):
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(uploaded_pop, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
                custom_pop_path = temp_dir
            elif uploaded_pop.name.endswith('.csv'):
                temp_dir = tempfile.mkdtemp()
                pop_csv_path = os.path.join(temp_dir, uploaded_pop.name)
                with open(pop_csv_path, "wb") as f:
                    f.write(uploaded_pop.getbuffer())
                custom_pop_path = pop_csv_path
        if uploaded_point is not None:
            temp_dir2 = tempfile.mkdtemp()
            point_csv_path = os.path.join(temp_dir2, uploaded_point.name)
            with open(point_csv_path, "wb") as f:
                f.write(uploaded_point.getbuffer())
            custom_point_path = point_csv_path

    st.sidebar.markdown("## Vulnerability Index Weights", unsafe_allow_html=True)
    st.sidebar.markdown("<small>All weights must sum to 1.0</small>", unsafe_allow_html=True)
    colw1, colw2 = st.sidebar.columns(2)
    with colw1:
        w1 = st.number_input("LST", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="w1")
        w3 = st.number_input("Children", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w3")
    with colw2:
        w2 = st.number_input("Elderly", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w2")
        w4 = st.number_input("Income", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="w4")
    total_weight = w1 + w2 + w3 + w4
    if abs(total_weight - 1.0) > 1e-6:
        st.sidebar.error("The sum of weights must be 1.0.")

    # ------------------- 页面导航与内容 -------------------
    section_names = [
        "Intro",
        "Model",
        "Vulnerability Index",
        "Conclusion",
    ]
    num_pages = len(section_names)
    if 'page_number' not in st.session_state:
        st.session_state['page_number'] = 1

    for i, name in enumerate(section_names):
        selected = (st.session_state['page_number'] == i + 1)
        if st.sidebar.button(name, key=f"sidebar_page_{i+1}", help=name):
            st.session_state['page_number'] = i + 1

    # 只保留页面内容函数调用，不再显示上下箭头
    if st.session_state['page_number'] == 1:
        page1()
    elif st.session_state['page_number'] == 2:
        page2()
    elif st.session_state['page_number'] == 3:
        page3()
    elif st.session_state['page_number'] == 4:
        page4()


# Streamlit app entry point
if __name__ == "__main__":
    main()