# Medicaid-Enrollment-Analysis

The purpose of this analysis is to explore Medicaid and CHIP enrollment data from 2013 to 2025 for all 50 states and Washington D.C. This code removes unnecessary columns and creates three exploratory visualizations of the data. 

## Description

The following steps were conducted within this analysis: 
1. Filtered data to only include "updated" records (as opposed to updated and preliminary).
2. Order data rows by state name and reporting period so that they are chronological.
3. Decipher the reporting period identifier to determine the fiscal year and month in which the record was uploaded for, for each state.
4. Determine how many records were submitted by each state within each fiscal year
5. Plot a line graph showing the fluctuations in median total enrollment over time for both Medicaid and CHIP in Washington D.C.
6. Create a dumbbell plot comparing the total Medicaid enrollment values for Mid-Atlantic states from December 2019 to December 2024.
7. Create a line graph showing the medican enrollment of CHIP and Medicaid enrollment for Mid-Atlantic States from 2013 to 2024.

## Getting Started

### Dependencies
To download the most up-to-date Medicaid data, go to: https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360/data

Relevant packages: 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import textwrap

### Installing

Modify file path names for uploading the Medicaid dataset and output files. 


## Author
Emily Quick-Cole

## Figures 
<img width="640" height="480" alt="Fig1" src="https://github.com/user-attachments/assets/9874f8bb-dfe1-40e5-89c0-ccddaeefb3a9" />
<img width="1000" height="600" alt="Fig2" src="https://github.com/user-attachments/assets/1203e567-dab5-456e-965a-a8dcbfca700a" />
<img width="812" height="457" alt="Fig3" src="https://github.com/user-attachments/assets/86eb6dd9-6c5d-4280-9c35-f5b82781762e" />
