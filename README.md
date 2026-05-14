# Medicaid-Enrollment-Analysis

The purpose of this analysis is to explore Medicaid and CHIP enrollment data from 2013 to 2025 for all 50 states and Washington D.C. This code removes unnecessary columns and creates three exploratory visualizations of the data. 

## Description

The following steps were conducted within this analysis: 
1. Set up a log file to capture data steps, checks, and flags.
2. Filtered data to only include "updated" records (as opposed to updated and preliminary).
3. Order data rows by state name and reporting period so that they are chronological.
4. Decipher the reporting period identifier to determine the fiscal year and month in which the record was uploaded for, for each state.
5. Determine how many records were submitted by each state within each fiscal year
6. Plot a line graph showing the fluctuations in median total enrollment over time for both Medicaid and CHIP in Washington D.C.
7. Create a dumbbell plot comparing the total Medicaid enrollment values for Mid-Atlantic states from December 2019 to December 2024.
8. Create a line graph showing the medican enrollment of CHIP and Medicaid enrollment for Mid-Atlantic States from 2013 to 2024.

## Getting Started

### Dependencies
To download the most up-to-date Medicaid data, go to: https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360/data
Or use the "MedicaidChipApps.EligibilityDet.Enroll.9.6.2025.csv"

Use "Medicaid Analysis.py" to see data cleaning and visualizations code.

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

## Log File
<details>
<summary>View Build Logs</summary>

```text
2026-05-14 13:48:53 | INFO | ============================================================
2026-05-14 13:48:53 | INFO | Medicaid Data Analyzer — Run started
2026-05-14 13:48:53 | INFO | ============================================================
2026-05-14 13:48:53 | INFO | Loading data from: /Users/emilyquick-cole/Downloads/MedicaidChipApps.EligibilityDet.Enroll.9.6.2025.csv
2026-05-14 13:48:53 | INFO | Data loaded successfully — 9792 rows, 44 columns
2026-05-14 13:48:53 | INFO | --- Data Exploration ---
2026-05-14 13:48:53 | INFO | Missing value check complete — columns with missing values:
New Applications Submitted to Medicaid and CHIP Agencies                                        51
New Applications Submitted to Medicaid and CHIP Agencies - footnotes                          7461
Applications for Financial Assistance Submitted to the State Based Marketplace                  51
Applications for Financial Assistance Submitted to the State Based Marketplace - footnotes    9537
Total Applications for Financial Assistance Submitted at State Level                            51
Total Applications for Financial Assistance Submitted at State Level - footnotes              7156
Individuals Determined Eligible for Medicaid at Application                                     51
Individuals Determined Eligible for Medicaid at Application - footnotes                       7100
Individuals Determined Eligible for CHIP at Application                                         51
Individuals Determined Eligible for CHIP at Application - footnotes                           8913
Total Medicaid and CHIP Determinations                                                          51
Total Medicaid and CHIP Determinations - footnotes                                            7968
Medicaid and CHIP Child Enrollment                                                              51
Medicaid and CHIP Child Enrollment - footnotes                                                9043
Total Medicaid and CHIP Enrollment                                                               2
Total Medicaid and CHIP Enrollment - footnotes                                                9141
Total Medicaid Enrollment                                                                       51
Total Medicaid Enrollment - footnotes                                                         9151
Total CHIP Enrollment                                                                           51
Total CHIP Enrollment - footnotes                                                             9431
Total Adult Medicaid Enrollment                                                               8721
Total Adult Medicaid Enrollment - footnotes                                                   9723
Total Medicaid and CHIP Determinations Processed in Less than 24 Hours                        4437
Total Medicaid and CHIP Determinations Processed in Less than 24 Hours - footnotes            8751
Total Medicaid and CHIP Determinations Processed Between 24 Hours and 7 Days                  4437
Total Medicaid and CHIP Determinations Processed Between 24 Hours and 7 Days - footnotes      8751
Total Medicaid and CHIP Determinations Processed Between 8 Days and 30 Days                   4437
Total Medicaid and CHIP Determinations Processed Between 8 Days and 30 Days - footnotes       8757
Total Medicaid and CHIP Determinations Processed between 31 days and 45 days                  4437
Total Medicaid and CHIP Determinations Processed between 31 days and 45 days - footnotes      8757
Total Medicaid and CHIP Determinations Processed in More than 45 Days                         4437
Total Medicaid and CHIP Determinations Processed in More than 45 Days - footnotes             8757
Total Call Center Volume (Number of Calls)                                                    7149
Total Call Center Volume (Number of Calls) - footnotes                                        7260
Average Call Center Wait Time (Minutes)                                                       7147
Average Call Center Wait Time (Minutes) - footnotes                                           7154
Average Call Center Abandonment Rate                                                          7149
Average Call Center Abandonment Rate - footnotes                                              7154
dtype: int64
2026-05-14 13:48:53 | INFO | --- Dropping footnote columns ---
2026-05-14 13:48:53 | INFO | Footnote columns identified for removal: ['New Applications Submitted to Medicaid and CHIP Agencies - footnotes', 'Applications for Financial Assistance Submitted to the State Based Marketplace - footnotes', 'Total Applications for Financial Assistance Submitted at State Level - footnotes', 'Individuals Determined Eligible for Medicaid at Application - footnotes', 'Individuals Determined Eligible for CHIP at Application - footnotes', 'Total Medicaid and CHIP Determinations - footnotes', 'Medicaid and CHIP Child Enrollment - footnotes', 'Total Medicaid and CHIP Enrollment - footnotes', 'Total Medicaid Enrollment - footnotes', 'Total CHIP Enrollment - footnotes', 'Total Adult Medicaid Enrollment - footnotes', 'Total Medicaid and CHIP Determinations Processed in Less than 24 Hours - footnotes', 'Total Medicaid and CHIP Determinations Processed Between 24 Hours and 7 Days - footnotes', 'Total Medicaid and CHIP Determinations Processed Between 8 Days and 30 Days - footnotes', 'Total Medicaid and CHIP Determinations Processed between 31 days and 45 days - footnotes', 'Total Medicaid and CHIP Determinations Processed in More than 45 Days - footnotes', 'Total Call Center Volume (Number of Calls) - footnotes', 'Average Call Center Wait Time (Minutes) - footnotes', 'Average Call Center Abandonment Rate - footnotes']
2026-05-14 13:48:53 | INFO | Footnote columns dropped — 25 columns remaining
2026-05-14 13:48:53 | INFO | --- Filtering to Updated records ---
2026-05-14 13:48:53 | INFO | "Preliminary or Updated" column converted to string dtype
2026-05-14 13:48:53 | INFO | Filter applied — 4896 Updated records retained, 4896 non-Updated records excluded
2026-05-14 13:48:53 | INFO | No duplicate rows found
2026-05-14 13:48:54 | INFO | Updated dataframe exported to updated_dataframe.xlsx
2026-05-14 13:48:54 | INFO | --- Sorting data by State Name and Reporting Period ---
2026-05-14 13:48:54 | INFO | Data sorted successfully by State Name and Reporting Period.
2026-05-14 13:48:54 | INFO | Sorted dataframe exported to sorted_dataframe.xlsx
2026-05-14 13:48:54 | INFO | --- Parsing Reporting Period into Fiscal Year and Month columns ---
2026-05-14 13:48:54 | INFO | Fiscal Year range: 2013 to 2025
2026-05-14 13:48:54 | INFO | Months present: ['09', '06', '07', '08', '10', '11', '12', '01', '02', '03', '04', '05']
2026-05-14 13:48:54 | INFO | --- Generating frequency tables ---
2026-05-14 13:48:54 | INFO | State by FY frequency table generated — 51 states across 10 fiscal years
2026-05-14 13:48:54 | INFO | 
--- Cross-Tabulation Log ---
Reporting Period Fiscal Years  2013  2017  2018  2019  2020  2021  2022  2023  2024  2025
State Name                                                                               
Alabama                           1     7    12    12    12    12    12    12    12     4
Alaska                            1     7    12    12    12    12    12    12    12     4
Arizona                           1     7    12    12    12    12    12    12    12     4
Arkansas                          1     7    12    12    12    12    12    12    12     4
California                        1     7    12    12    12    12    12    12    12     4
Colorado                          1     7    12    12    12    12    12    12    12     4
Connecticut                       1     7    12    12    12    12    12    12    12     4
Delaware                          1     7    12    12    12    12    12    12    12     4
District of Columbia              1     7    12    12    12    12    12    12    12     4
Florida                           1     7    12    12    12    12    12    12    12     4
Georgia                           1     7    12    12    12    12    12    12    12     4
Hawaii                            1     7    12    12    12    12    12    12    12     4
Idaho                             1     7    12    12    12    12    12    12    12     4
Illinois                          1     7    12    12    12    12    12    12    12     4
Indiana                           1     7    12    12    12    12    12    12    12     4
Iowa                              1     7    12    12    12    12    12    12    12     4
Kansas                            1     7    12    12    12    12    12    12    12     4
Kentucky                          1     7    12    12    12    12    12    12    12     4
Louisiana                         1     7    12    12    12    12    12    12    12     4
Maine                             1     7    12    12    12    12    12    12    12     4
Maryland                          1     7    12    12    12    12    12    12    12     4
Massachusetts                     1     7    12    12    12    12    12    12    12     4
Michigan                          1     7    12    12    12    12    12    12    12     4
Minnesota                         1     7    12    12    12    12    12    12    12     4
Mississippi                       1     7    12    12    12    12    12    12    12     4
Missouri                          1     7    12    12    12    12    12    12    12     4
Montana                           1     7    12    12    12    12    12    12    12     4
Nebraska                          1     7    12    12    12    12    12    12    12     4
Nevada                            1     7    12    12    12    12    12    12    12     4
New Hampshire                     1     7    12    12    12    12    12    12    12     4
New Jersey                        1     7    12    12    12    12    12    12    12     4
New Mexico                        1     7    12    12    12    12    12    12    12     4
New York                          1     7    12    12    12    12    12    12    12     4
North Carolina                    1     7    12    12    12    12    12    12    12     4
North Dakota                      1     7    12    12    12    12    12    12    12     4
Ohio                              1     7    12    12    12    12    12    12    12     4
Oklahoma                          1     7    12    12    12    12    12    12    12     4
Oregon                            1     7    12    12    12    12    12    12    12     4
Pennsylvania                      1     7    12    12    12    12    12    12    12     4
Rhode Island                      1     7    12    12    12    12    12    12    12     4
South Carolina                    1     7    12    12    12    12    12    12    12     4
South Dakota                      1     7    12    12    12    12    12    12    12     4
Tennessee                         1     7    12    12    12    12    12    12    12     4
Texas                             1     7    12    12    12    12    12    12    12     4
Utah                              1     7    12    12    12    12    12    12    12     4
Vermont                           1     7    12    12    12    12    12    12    12     4
Virginia                          1     7    12    12    12    12    12    12    12     4
Washington                        1     7    12    12    12    12    12    12    12     4
West Virginia                     1     7    12    12    12    12    12    12    12     4
Wisconsin                         1     7    12    12    12    12    12    12    12     4
Wyoming                           1     7    12    12    12    12    12    12    12     4
----------------------------
2026-05-14 13:48:54 | INFO | State by Month frequency table generaged - 51 states across 12 months.
2026-05-14 13:48:54 | INFO | 
--- Cross-Tabulation Log ---
Reporting Period Month  01  02  03  04  05  06  07  08  09  10  11  12
State Name                                                            
Alabama                  8   8   8   8   7   8   8   8   9   8   8   8
Alaska                   8   8   8   8   7   8   8   8   9   8   8   8
Arizona                  8   8   8   8   7   8   8   8   9   8   8   8
Arkansas                 8   8   8   8   7   8   8   8   9   8   8   8
California               8   8   8   8   7   8   8   8   9   8   8   8
Colorado                 8   8   8   8   7   8   8   8   9   8   8   8
Connecticut              8   8   8   8   7   8   8   8   9   8   8   8
Delaware                 8   8   8   8   7   8   8   8   9   8   8   8
District of Columbia     8   8   8   8   7   8   8   8   9   8   8   8
Florida                  8   8   8   8   7   8   8   8   9   8   8   8
Georgia                  8   8   8   8   7   8   8   8   9   8   8   8
Hawaii                   8   8   8   8   7   8   8   8   9   8   8   8
Idaho                    8   8   8   8   7   8   8   8   9   8   8   8
Illinois                 8   8   8   8   7   8   8   8   9   8   8   8
Indiana                  8   8   8   8   7   8   8   8   9   8   8   8
Iowa                     8   8   8   8   7   8   8   8   9   8   8   8
Kansas                   8   8   8   8   7   8   8   8   9   8   8   8
Kentucky                 8   8   8   8   7   8   8   8   9   8   8   8
Louisiana                8   8   8   8   7   8   8   8   9   8   8   8
Maine                    8   8   8   8   7   8   8   8   9   8   8   8
Maryland                 8   8   8   8   7   8   8   8   9   8   8   8
Massachusetts            8   8   8   8   7   8   8   8   9   8   8   8
Michigan                 8   8   8   8   7   8   8   8   9   8   8   8
Minnesota                8   8   8   8   7   8   8   8   9   8   8   8
Mississippi              8   8   8   8   7   8   8   8   9   8   8   8
Missouri                 8   8   8   8   7   8   8   8   9   8   8   8
Montana                  8   8   8   8   7   8   8   8   9   8   8   8
Nebraska                 8   8   8   8   7   8   8   8   9   8   8   8
Nevada                   8   8   8   8   7   8   8   8   9   8   8   8
New Hampshire            8   8   8   8   7   8   8   8   9   8   8   8
New Jersey               8   8   8   8   7   8   8   8   9   8   8   8
New Mexico               8   8   8   8   7   8   8   8   9   8   8   8
New York                 8   8   8   8   7   8   8   8   9   8   8   8
North Carolina           8   8   8   8   7   8   8   8   9   8   8   8
North Dakota             8   8   8   8   7   8   8   8   9   8   8   8
Ohio                     8   8   8   8   7   8   8   8   9   8   8   8
Oklahoma                 8   8   8   8   7   8   8   8   9   8   8   8
Oregon                   8   8   8   8   7   8   8   8   9   8   8   8
Pennsylvania             8   8   8   8   7   8   8   8   9   8   8   8
Rhode Island             8   8   8   8   7   8   8   8   9   8   8   8
South Carolina           8   8   8   8   7   8   8   8   9   8   8   8
South Dakota             8   8   8   8   7   8   8   8   9   8   8   8
Tennessee                8   8   8   8   7   8   8   8   9   8   8   8
Texas                    8   8   8   8   7   8   8   8   9   8   8   8
Utah                     8   8   8   8   7   8   8   8   9   8   8   8
Vermont                  8   8   8   8   7   8   8   8   9   8   8   8
Virginia                 8   8   8   8   7   8   8   8   9   8   8   8
Washington               8   8   8   8   7   8   8   8   9   8   8   8
West Virginia            8   8   8   8   7   8   8   8   9   8   8   8
Wisconsin                8   8   8   8   7   8   8   8   9   8   8   8
Wyoming                  8   8   8   8   7   8   8   8   9   8   8   8
----------------------------
2026-05-14 13:48:55 | INFO | Sorted dataframe and frequency tables exported to sorted_dataframe.xlsx
2026-05-14 13:48:55 | INFO | --- Figure 1: DC Medicaid and CHIP enrollment line plot ---
2026-05-14 13:48:55 | INFO | District of Columbia records confirmed present for Figure 1
2026-05-14 13:48:55 | INFO | 
--- Data for Visual 1 ---
  Reporting Period Fiscal Year  Total Medicaid Enrollment  Total CHIP Enrollment
0                         2013                        NaN                    NaN
1                         2017                   244961.0                13226.0
2                         2018                   239295.5                15554.5
3                         2019                   237543.0                17104.5
4                         2020                   235651.5                17372.0
5                         2021                   253055.5                17236.5
6                         2022                   269313.0                17023.0
7                         2023                   277120.0                16530.0
8                         2024                   244294.5                17381.5
9                         2025                   243355.0                17458.5
----------------------------
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Figure 1 saved to /Users/emilyquick-cole/Documents/Python/medicaid_data/Fig1.png
2026-05-14 13:48:55 | INFO | --- Figure 2: Mid-Atlantic dumbbell chart ---
2026-05-14 13:48:55 | INFO | Dumbbell chart data — 8 states, 16 total records
2026-05-14 13:48:55 | INFO | There are 8 states and 16 datapoints in the filtered data, which is what is expected.
2026-05-14 13:48:55 | INFO | 
--- Data for Visual 2 ---
                State Name Reporting Period Fiscal Year Reporting Period Month  Total Medicaid Enrollment
1598              Delaware                         2019                     12                   218227.0
1718              Delaware                         2024                     12                   236840.0
1406  District of Columbia                         2019                     12                   235995.0
1526  District of Columbia                         2024                     12                   242560.0
3902              Maryland                         2019                     12                  1187568.0
4022              Maryland                         2024                     12                  1291976.0
6014            New Jersey                         2019                     12                  1476344.0
6134            New Jersey                         2024                     12                  1543150.0
6590              New York                         2019                     12                  5394013.0
6710              New York                         2024                     12                  5974991.0
7358          Pennsylvania                         2019                     12                  2685303.0
7478          Pennsylvania                         2024                     12                  2800122.0
8702              Virginia                         2019                     12                  1258915.0
8822              Virginia                         2024                     12                  1609622.0
9470         West Virginia                         2019                     12                   479633.0
9590         West Virginia                         2024                     12                   467632.0
----------------------------
2026-05-14 13:48:55 | INFO | Figure 2 saved to /Users/emilyquick-cole/Documents/Python/medicaid_data/Fig2.png
2026-05-14 13:48:55 | INFO | --- Figure 3: Mid-Atlantic CHIP and Medicaid line plot ---
2026-05-14 13:48:55 | INFO | Figure 3 data — 8 Mid-Atlantic states, 80 records
2026-05-14 13:48:55 | INFO | 
--- Data for Visual 3 ---
               State Name Reporting Period Fiscal Year  Total Medicaid and CHIP Enrollment
70               Delaware                         2013                            223324.0
71               Delaware                         2017                            230339.0
72               Delaware                         2018                            231935.5
73               Delaware                         2019                            231592.0
74               Delaware                         2020                            241912.5
75               Delaware                         2021                            270061.0
76               Delaware                         2022                            290005.5
77               Delaware                         2023                            304340.0
78               Delaware                         2024                            251992.0
79               Delaware                         2025                            247615.0
80   District of Columbia                         2013                            235786.0
81   District of Columbia                         2017                            258187.0
82   District of Columbia                         2018                            254850.0
83   District of Columbia                         2019                            254201.0
84   District of Columbia                         2020                            253137.0
85   District of Columbia                         2021                            270264.5
86   District of Columbia                         2022                            286336.0
87   District of Columbia                         2023                            293473.0
88   District of Columbia                         2024                            261743.0
89   District of Columbia                         2025                            260799.5
200              Maryland                         2013                            856297.0
201              Maryland                         2017                           1309986.0
202              Maryland                         2018                           1315352.5
203              Maryland                         2019                           1327102.0
204              Maryland                         2020                           1384795.0
205              Maryland                         2021                           1528983.5
206              Maryland                         2022                           1641160.0
207              Maryland                         2023                           1696070.5
208              Maryland                         2024                           1586716.5
209              Maryland                         2025                           1457030.5
300            New Jersey                         2013                           1283851.0
301            New Jersey                         2017                           1774040.0
302            New Jersey                         2018                           1780430.5
303            New Jersey                         2019                           1723564.0
304            New Jersey                         2020                           1794715.0
305            New Jersey                         2021                           2000684.5
306            New Jersey                         2022                           2143899.0
307            New Jersey                         2023                           2242288.0
308            New Jersey                         2024                           1831085.5
309            New Jersey                         2025                           1806682.5
320              New York                         2013                           5678417.0
321              New York                         2017                           6089591.0
322              New York                         2018                           6140386.5
323              New York                         2019                           6091164.0
324              New York                         2020                           6314776.5
325              New York                         2021                           6892509.5
326              New York                         2022                           7237145.5
327              New York                         2023                           7484273.5
328              New York                         2024                           6713890.0
329              New York                         2025                           6613952.0
380          Pennsylvania                         2013                           2386046.0
381          Pennsylvania                         2017                           3013365.0
382          Pennsylvania                         2018                           3035263.5
383          Pennsylvania                         2019                           3007938.5
384          Pennsylvania                         2020                           3134654.5
385          Pennsylvania                         2021                           3414320.5
386          Pennsylvania                         2022                           3613738.5
387          Pennsylvania                         2023                           3669297.0
388          Pennsylvania                         2024                           3126333.5
389          Pennsylvania                         2025                           3104997.5
460              Virginia                         2013                            935434.0
461              Virginia                         2017                           1026222.0
462              Virginia                         2018                           1045207.0
463              Virginia                         2019                           1330564.5
464              Virginia                         2020                           1518790.0
465              Virginia                         2021                           1741254.5
466              Virginia                         2022                           1930032.0
467              Virginia                         2023                           2022459.5
468              Virginia                         2024                           1842993.5
469              Virginia                         2025                           1796660.5
480         West Virginia                         2013                            354544.0
481         West Virginia                         2017                            544461.0
482         West Virginia                         2018                            531772.5
483         West Virginia                         2019                            516481.5
484         West Virginia                         2020                            535806.0
485         West Virginia                         2021                            591862.0
486         West Virginia                         2022                            629993.5
487         West Virginia                         2023                            604511.0
488         West Virginia                         2024                            513675.0
489         West Virginia                         2025                            506745.5
----------------------------
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2026-05-14 13:48:55 | INFO | Figure 3 saved to /Users/emilyquick-cole/Documents/Python/medicaid_data/Fig3.png
2026-05-14 13:48:55 | INFO | ============================================================
2026-05-14 13:48:55 | INFO | Medicaid Data Analyzer — Run complete
2026-05-14 13:48:55 | INFO | ============================================================

```

</details>
