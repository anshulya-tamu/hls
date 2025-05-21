# **SARAH-CONUS: Sub-weekly Area of Reservoirs from Analysis of Harmonized Landsat and Sentinel-2 data for Continental US** 

<p align="justify">
Welcome to the official repository for our sub-weekly reservoir surface area mapping algorithm. This project focuses on applying a <strong>Random Forest (RF) classification model</strong> and a <strong>refined image enhancement algorithm</strong> to <strong>Harmonized Landsat-Sentinel (HLS)</strong> imagery, producing <strong>sub-weekly</strong> surface area time series for thousands of reservoirs in the <strong>Continental United States (CONUS)</strong>.
</p>

## **Table of Contents**

1. [Project Description](#project-description)  
2. [Features / Pipeline Overview](#features--pipeline-overview)  
3. [Prerequisites & Installation](#prerequisites--installation)  
4. [Quick Start](#quick-start)  
5. [Detailed Usage](#detailed-usage) 
6. [Example](#example)  
7. [Project Structure](#project-structure)  
8. [Data Sources](#data-sources)  
9. [License](#license)  
10. [Citation & References](#citation--references)  
11. [Contact & Contributing](#contact--contributing)

## **Project Description**

This project aims to generate **sub-weekly reservoir surface area time series** for reservoirs across the Continental United States. By using **Harmonized Landsat and Sentinel-2 (HLS)** datasets, we can overcome the trade-off between **spatial resolution (30 m)** and **temporal frequency (~2–6 days)**. Our approach includes:

- **Random Forest Classification**: Classify each pixel in HLS data into *Water, Land, Cloud,* or *Ice*.  
- **Enhanced Image Correction**: Address cloud contamination and mixed-pixel misclassifications using a physics-based enhancement algorithm.  
- **Sub-Weekly Time Series**: Merge classification outputs over time to form near-real-time reservoir surface area time series.  
- **LOWESS Gap-Filling**: Interpolate or smooth out missing or outlier data points where reservoir imagery is unusable (>95% cloud cover).

**Key Findings**  
- Achieves R² = 0.98 and bias < 10% when compared to in-situ data from 240 reservoirs.  
- Offers **sub-weekly** temporal resolution, providing richer intra-monthly dynamics than typical monthly products.  
- Demonstrates utility for water resource management, hydropower, flood risk analysis, and broader hydrological studies.

## **Features / Pipeline Overview**

1. **Reservoir Identification**  
   - Leverages the [Global Reservoir and Dam (GRanD) v1.3](https://globaldamwatch.org/grand/) dataset to locate ~1,900 CONUS reservoirs of various sizes, shapes, and climatic zones.

2. **HLS Data Acquisition**  
   - Downloads the **HLS (v1.4) surface reflectance** product from NASA’s LP-DAAC for 2016–2023.  
   - Tiles are automatically merged if a reservoir crosses multiple HLS tiles.

3. **Random Forest Classification**  
   - Trained on a labeled set of ~3.48 million pixels spanning *water, land, cloud,* and *ice* classes.  
   - Incorporates spectral bands (e.g., NIR, SWIR), water indices (NDWI, NDMI, etc.), and 3×3 spatial averages.

4. **Refined Image Enhancement**  
   - Cloud/mixed-pixel correction using a dynamic threshold approach based on **water occurrence data** (Pekel et al., 2016).  
   - Identifies edge pixels in the reservoir to extrapolate the *true* water boundary more accurately.

5. **Time Series Extraction**  
   - Computes surface area by counting classified “water” pixels and converting to km².  
   - Excludes days when >95% of the reservoir area is cloud-covered.

6. **LOWESS Smoothing and Outlier Removal**  
   - Fills gaps on cloudy days and filters spurious outliers.  
   - Ensures a continuous sub-weekly surface area record.

7. **Validation**  
   - Compared against daily in-situ reservoir areas at 240 sites.  
   - Demonstrates robust performance across reservoir size classes and climatic regimes.

## **Prerequisites & Installation**

1. **Software Requirements**  
   - Python 3.8+  
   - GDAL (for geospatial data processing)  
   - A modern OS (Windows, macOS, Linux)

2. **Libraries**  
   - `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `rasterio`, `statsmodels` (for LOWESS), etc.  
   - (Optional) `jupyter` for viewing example notebooks.

3. **Installation**  
   ```bash
   # Clone this repository
   git clone https://github.com/anshulya-tamu/hls.git
   cd hls

   # Install dependencies (conda recommended)
   pip install -r requirements.txt

## **Example**

A more detailed demonstration is provided in the examples/ folder:
1. `examples/single-reservoir-example.ipynb`  
    - Walks through data download, classification, enhancement, and final time series plotting for a single small reservoir.  
    - Illustrates how to handle special cases (e.g., ice coverage, ephemeral water bodies).

**PLACEHOLDER FOR FIGURE**  
_An example plot showing the time series of reservoir surface area with LOWESS smoothing._

## **Data Sources**

1. **HLS Data**:  
   - Harmonized Landsat and Sentinel-2 (HLS) L30 & S30 images from [NASA LP-DAAC](https://lpdaac.usgs.gov/).  
   - Product version used: **v1.4** (2016–2023).  
2. **Reservoir Masks**:  
   - [Global Reservoir and Dam (GRanD) v1.3](https://globaldamwatch.org/grand/) for reservoir polygons.  
3. **Auxiliary Water Occurrence**:  
   - [Global Surface Water Dataset (GSWD)](https://global-surface-water.appspot.com/) by Pekel et al. (2016).  
4. **In-Situ Data**:  
   - Ground-based reservoir surface areas from 240 CONUS reservoirs for validation.

## **License**

We recommend using an open-source license such as **MIT** or **Apache 2.0** to facilitate collaboration and reproducibility:

## **Citation & References**

If you find this code or dataset helpful, please cite our work:

> Yadav, A., Zhang, S., Zhao, B., Allen, G. H., Pearson, C., Huntington, J., et al. (2025). Mapping reservoir water surface area in the contiguous United States using the high‐temporal Harmonized Landsat and Sentinel (HLS) data at a sub‐weekly time scale. Geophysical Research Letters, 52, e2024GL114046. https://doi.org/10.1029/2024GL114046
---
## **Contact & Contributing**

1. **Contact**  
   - [Anshul Yadav, Texas A&M University]  
   - [anshulya@tamu.edu]

2. **How to Contribute**  
   - Fork this repo and create a feature branch.  
   - Open an issue if you encounter a bug or have a feature request.  
   - Submit a pull request once you’ve tested your changes.

3. **Acknowledgments**  
   - This research was supported by the NASA Earth Sciences, Applied Sciences - Water Resources Program (80NSSC22K0933), U.S. Geological Survey Water Availability and Use Science Program and Water Resources Research Act Program (G25AC00180-00). 
   - Thanks to the Bureau of Reclamation, Desert Research Institute, and all collaborators for in-situ data support.
---
<p align="center">
<i>Thank you for using our sub-weekly reservoir area monitoring code!
We hope it aids your hydrological research and reservoir management endeavors.</i>
</p>
