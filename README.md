# Deep Learning for Urban Land Cover Classification and Land Value Prediction - Londrina-Maringá Metropolitan Region

## Overview

This repository hosts interactive thematic maps displaying urban land cover classification and land value predictions for the 12 municipalities that compose the Londrina-Maringá Metropolitan Region in northern Paraná State, southern Brazil, based on deep learning approaches.

## Study Area

The analysis encompasses municipalities in the Londrina-Maringá Metropolitan Region, including:

- Apucarana
- Arapongas
- Cambé
- Cambira
- Ibiporã
- Jandaia do Sul
- Londrina
- Mandaguari
- Maringá
- Marialva
- Rolândia
- Sarandi

## Project Description

This project provides spatial visualizations of:

### 1. Land Cover Classification

Classification of urban land cover into 10 custom categories:
- Bare Soil
- Shrub/Scrub
- Row Crops
- Grass/Pasture
- Developed High Density
- Developed Industrial
- Developed Low Density
- Developed Medium Density
- Perennial Trees
- Perennial Water

### 2. Land Value Analysis

Estimation of urban land unit values and associated confidence intervals.

## Technical Specifications

### Land Cover Classification (2024)
- **Spatial Resolution**: 109.45m × 109.45m grid cells
- **Coordinate System**: UTM Zone 22S (EPSG:32722)
- **Base Year**: 2024

### Land Value Analysis
**2024 Analysis**
- **Spatial Resolution**: 109.45m × 109.45m grid cells
- **Coordinate System**: UTM Zone 22S (EPSG:32722)
- **Reference Lot**: Unit values derived from a 363 m² vacant paradigm parcel

**2000-2021 Analysis**
- **Spatial Resolution**: 50m × 50m grid cells
- **Coordinate System**: UTM Zone 22S (EPSG:32722)
- **Reference Lot**: Unit values derived from a 363 m² vacant, flat parcel on paved street paradigm 

## Interactive Maps

Access the interactive maps at: [https://fjrcosta.github.io/landCoverlandValue/](https://fjrcosta.github.io/landCoverlandValue/)

## Methodology

### Land Cover Classification

The urban land cover classification leverages image encoders from deep learning models fine-tuned on custom-extracted satellite imagery, pioneering its application in Brazil for custom urban land cover analysis.

**Key Features**:
- 10 land cover categories including varying urban densities
- Utilizes freely available high-resolution remote sensing imagery
- Adaptable framework for user-defined categories
- Models: CLIP fine-tuned on the RSICD dataset, and DINOv2 fine-tuned using LoRA

### Land Value Prediction

**2024 Analysis**

The urban land value predictions for 2024 employ locational descriptors derived from remote sensing imagery (RSI) and learned embeddings, moving beyond traditional point-of-interest (POI) based approaches. The methodology is grounded in the concept of *agnostic location features*—descriptors that are independent of local regulations and globally accessible.

**Key Contributions**:
- Formalization of *agnostic location features* as descriptors of spatial context independent of local regulations
- Spatial modeling using a basis function constructed from the tensor product of thin-plate splines
- Model-based urban land value mapping using vacant parcel unit prices as proxy for locational value
- Baseline model: TabPFN v2 (Tabular Prior-Fitting Networks) with spatial cross-validation for transferability assessment

**2000-2021 Analysis**

Genuinely spatiotemporal modeling framework for the period 2000-2021 using GAMLSS (Generalized Additive Models for Location, Scale and Shape).

**Key Features**:
- Use of a basis function constructed from the tensor product of a spatial basis (thin-plate splines) and a temporal basis (splines)
- Spatiotemporal evolution of urban land values

## Research Team

**Institution**: Department of Statistics, Londrina State University (UEL), Brazil

**Author**  
Prof. MSc Eng. Felinto Costa  
Email: fjrcosta@uel.br

**PhD Advisor**  
Prof. Dr.   
Email: @uel.br

**Collaborator**  
Prof. Dr.  
Email: @uel.br

## License

This project is made available for academic and research purposes.

## Citation

If you use these maps or data in your research, please cite appropriately and contact the author for proper attribution.

## Acknowledgments

This work builds upon advanced deep learning frameworks:
- **Land Cover Classification**: CLIP fine-tuned on the RSICD dataset, and DINOv2 fine-tuned using LoRA on custom-extracted satellite imagery
- **Land Value Analysis**: TabPFN v2 trained on the BRPR13K dataset of urban land transactions in Paraná, Brazil, GAMLSS framework for spatiotemporal modeling, and foundation models (AlphaEarth) for learned spatial embeddings
