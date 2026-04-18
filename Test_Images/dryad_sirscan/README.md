# README

## Title

Image Dataset of Disk Diffusion Assay Scanned with the SIRscan System

## Citation

Egli, Adrian et al. (2023). Image Dataset of Disk Diffusion Assay Scanned with the SIRscan System. Dryad Digital Repository. [https://doi.org/10.5061/dryad.5dv41nsfj](https://doi.org/10.5061/dryad.5dv41nsfj)

## Authors

* **Prof. Adrian Egli**
  * Institute of Medical Microbiology
  * University of Zurich
  * Email: [aegli@imm.uzh.ch](mailto:aegli@imm.uzh.ch)

## Contact Information

For any questions or further information regarding this dataset, please contact:

* **Prof. Adrian Egli**
  * Email: [aegli@imm.uzh.ch](mailto:aegli@imm.uzh.ch)

## Date of Data Collection

* January 2020 to December 2022

## Geographic Location of Data Collection

* Institute of Medical Microbiology
* University of Zurich
* Zurich, Switzerland

## Project Description

This dataset comprises images and measurement data from routine microbiological diagnostics using the disk diffusion assay to assess antimicrobial resistance (AMR) in bacterial isolates. The images were generated using the **SIRscan** system, an automated device for scanning and interpreting disk diffusion plates.

The disk diffusion assay was conducted according to the **European Committee on Antimicrobial Susceptibility Testing (EUCAST)** guidelines. The assay involves placing antibiotic-impregnated disks on agar plates inoculated with bacteria and measuring the inhibition zones to determine the susceptibility of bacteria to various antibiotics.

Based on the measured inhibition zones, bacteria are categorized as **Sensitive (S)**, **Intermediate (I)**, or **Resistant (R)** to each antibiotic tested. The dataset also includes information on specific resistance mechanisms inferred from the resistance patterns:

* **No Resistance Mechanism**: Bacteria are susceptible to tested antibiotics.
* **Extended Spectrum Beta-Lactamases (ESBL)**
* **AmpC Beta-Lactamases (AmpC)**
* **Carbapenemases**
* **Combination of Mechanisms**: Occurrence of multiple resistance mechanisms.

These resistance mechanisms may occur individually or in combination.

## Description of Data Files

The dataset is organized into three main compressed files:

### 1. `images_original.zip`

* **Description**: Contains the original, unprocessed images of disk diffusion plates captured by the SIRscan system.
* **File Format**: JPEG images (`.jpg`)
* **Number of Files**: 225 images
* **Folder Structure**:

  `images_original/ ├── Sample001_20200101_090000.jpg ├── Sample002_20200101_093000.jpg ├── ...`
* **File Naming Convention**:
  * Each image file is named using the format: `SampleID_Date_Time.jpg`, where:
    * **SampleID**: Unique identifier for the bacterial isolate (e.g., Sample001).
    * **Date**: Date of image capture in `YYYYMMDD` format.
    * **Time**: Time of image capture in `HHMMSS` format.

### 2. `images_measured.zip`

* **Description**: Contains images annotated with measured inhibition zones, where an AI system has recognized and measured the zones.
* **File Format**: JPEG images (`.jpg`) with annotations.
* **Number of Files**: 225 images
* **Folder Structure**:

  `images_measured/ ├── Sample001_20200101_090000_measured.jpg ├── Sample002_20200101_093000_measured.jpg ├── ...`
* **File Naming Convention**:
  * Each image file corresponds to the original image but with `_measured` appended to indicate the measurements.

### 3. `Tables.zip`

* **Description**: Contains tabular data with detailed measurements and interpretations for each antibiotic tested against each bacterial isolate.
* **File Format**: CSV files (`.csv`)
* **Number of Files**: 1 file
* **Folder Structure**:

  `Tables/ └── measurements.csv`
* **File Contents**:

  The `measurements.csv` file includes the following columns:
  * **SampleID**: Unique identifier for each bacterial isolate.
  * **Date**: Date of the test (`YYYY-MM-DD` format).
  * **Antibiotic**: Name of the antibiotic tested.
  * **ZoneDiameter_mm**: Measured diameter of the inhibition zone in millimeters.
  * **Interpretation**: Categorical assessment based on EUCAST guidelines (`S`, `I`, `R`).
  * **ResistanceMechanism**: Identified resistance mechanism(s) (`None`, `ESBL`, `AmpC`, `Carbapenemase`, `Combination`).
  * **Comments**: Additional notes or observations.

## File Organization and Structure

* The dataset is divided into three main directories, each compressed into a ZIP file for ease of distribution.
* **`images_original.zip`** and **`images_measured.zip`** contain image files organized in a flat structure without subfolders.
* **`Tables.zip`** contains the `measurements.csv` file with all measurement data.

## Variable Definitions

### Image Files

* **SampleID**: A unique alphanumeric code assigned to each bacterial isolate (e.g., `Sample001`).
* **Date**: Date when the image was captured (`YYYYMMDD` format).
* **Time**: Time when the image was captured (`HHMMSS` format).

### Measurements CSV File (`measurements.csv`)

* **SampleID**: Matches the `SampleID` in the image files; used to link measurement data with images.
* **Date**: Date of the disk diffusion assay (`YYYY-MM-DD` format).
* **Antibiotic**: The antibiotic agent tested (e.g., Amoxicillin, Ciprofloxacin).
* **ZoneDiameter_mm**: The measured diameter of the inhibition zone, in millimeters.
* **Interpretation**:
  * **S**: Sensitive
  * **I**: Intermediate
  * **R**: Resistant
* **ResistanceMechanism**:
  * **None**: No specific resistance mechanism detected.
  * **ESBL**: Presence of Extended Spectrum Beta-Lactamase.
  * **AmpC**: Presence of AmpC beta-lactamase.
  * **Carbapenemase**: Presence of carbapenemase enzymes.
  * **Combination**: Multiple resistance mechanisms present.
* **Comments**: Any additional information or observations noted during the assay.

## Methods

### Sample Collection

* Bacterial isolates were collected from patient samples submitted for routine diagnostic testing at the Institute of Medical Microbiology, University of Zurich.
* Isolates include a range of clinically relevant bacterial species.

### Disk Diffusion Assay

* Conducted according to **EUCAST guidelines** (version 2022).
* Mueller-Hinton agar plates were inoculated with bacterial suspensions adjusted to a 0.5 McFarland standard.
* Antibiotic disks were placed on the agar surface using standardized protocols.
* Plates were incubated at **35±1°C** for **16-20 hours**.

### Image Acquisition

* After incubation, plates were scanned using the **SIRscan 2000** automated system.
* Original images were saved in high-resolution JPEG format.

### Measurement and Interpretation

* Inhibition zones were measured using AI software integrated with the SIRscan system.
* The AI system automatically detects the edges of inhibition zones and calculates their diameters.
* Measurements were reviewed and validated by trained microbiologists.
* Interpretations (`S`, `I`, `R`) were assigned based on EUCAST breakpoint tables.

### Resistance Mechanism Identification

* Based on specific inhibition zone patterns and confirmatory tests, resistance mechanisms were inferred.
* For example, resistance to cephalosporins and susceptibility to carbapenems may indicate ESBL production.

## Software Requirements

* **Image Viewing**: Any standard image viewer capable of opening JPEG files (e.g., Windows Photo Viewer, macOS Preview).
* **Data Analysis**: Software capable of opening CSV files, such as Microsoft Excel, LibreOffice Calc, or statistical software like R or Python.

## Data Usage Notes

* Data is anonymized and does not contain any patient-identifiable information.
* Users should comply with data protection regulations when using the dataset.

## License and Access Restrictions

* This dataset is licensed under the **Creative Commons Attribution 0 (CC0)**. However, if possible we would appreciate a proper citation of this dataset.
* Users are free to share and adapt the material for any purpose, even commercially, provided appropriate credit is given.

## Acknowledgements

* We thank the staff of the Institute of Medical Microbiology, University of Zurich, for their assistance in data collection and validation.
* Special thanks to the technical team for their support with the SIRscan system.

## Funding Sources

* This work was supported by the **the University of Zurich**.

## Data Source

* The images and data were derived from routine diagnostic activities at the Institute of Medical Microbiology, University of Zurich.

---

**In Case of Questions:**

Please contact the main author, **Prof. Adrian Egli**, at **[aegli@uzh.ch](mailto:aegli@uzh.ch)** for any inquiries regarding the dataset.

---

## Summary

This README file provides a comprehensive overview of the dataset titled "Image Dataset of Disk Diffusion Assay Scanned with the SIRscan System." It includes detailed descriptions of the data files, their organization, variables, and the methods used to generate the data. The dataset is valuable for researchers interested in antimicrobial resistance, microbiology diagnostics, image analysis, and the development of AI tools in medical applications.

By providing this detailed README, we aim to ensure that anyone interested in reusing the data can easily understand and navigate the dataset, facilitating transparency and reproducibility in research.
