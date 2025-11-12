# Land Cover and Land Value Maps - Londrina Metropolitan Region

## Overview
This repository hosts interactive thematic maps displaying urban land cover classification and land value predictions for the Londrina Metropolitan Region in Paraná State, Brazil based on deep learning approaches including CLIP ViTB32 and TabPFN v2 transformer-based models.

## Project Description
This project provides spatial visualization of:
1. **Land Cover Classification**: Urban land cover types across the metropolitan region
2. **Land Value Analysis**: Predicted unit land values with confidence intervals

## Study Area
The analysis encompasses municipalities in the Londrina Metropolitan Region, including:
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

## Technical Specifications
- **Spatial Resolution**: 109.45m × 109.45m grid cells
- **Coordinate System**: UTM Zone 22S (EPSG:32722)
- **Reference Lot**: 363 m² paradigm lot
- **Base Year**: 2024

## Maps Available
Access the interactive maps at: [https://fjrcosta.github.io/landCoverlandValue/](https://fjrcosta.github.io/landCoverlandValue/)

## Methodology

### Land Cover Classification
The land cover classification leverages the image encoder from a fine-tuned Contrastive Language-Image Pre-Training (CLIP) model, pioneering its application in Brazil for urban land cover analysis. 

**Key Features**:
- 10 land cover categories including varying urban densities
- 93% classification accuracy
- Utilizes freely available high-resolution remote sensing imagery
- Adaptable framework for user-defined categories

### Land Value Prediction
The land value predictions employ data-driven locational descriptors derived from remote sensing imagery (RSI) and learned embeddings, moving beyond traditional point-of-interest (POI) based approaches. The methodology is grounded in the concept of *agnostic location features*—descriptors that are independent of local regulations and globally accessible. 

**Key Contributions**:
- Formalization of *agnostic location features* as descriptors of spatial context independent of local regulations
- Model-based urban land value mapping using vacant parcel unit prices as proxy for locational value
- Baseline model: TabPFN v2 (Tabular Prior-Fitting Networks) with spatial cross-validation for transferability assessment

## Contact
**Author**: Prof. MSc Eng. Felinto Costa  
**Institution**: Department of Statistics, Londrina State University (UEL)  
**Email**: fjrcosta@uel.br

## License
This project is made available for academic and research purposes.

## Citation
If you use these maps or data in your research, please cite appropriately and contact the author for proper attribution.

## Acknowledgments
This work builds upon advanced deep learning frameworks including CLIP for land cover classification and foundation models (AlphaEarth) for learned spatial embeddings. The land value analysis leverages the BRPR13K dataset of urban land transactions in Paraná, Brazil.
