''' Medicaid Data Analyzer
Date: September 6, 2025
Name: Emily Quick-Cole

Purpose: To explore and clean the Medicaid enrollment data Medicaid data. I specifically remove non-updated rows from
the data set, remove columns designated as containing "footnotes", convert the "Reporting Period" identifier into a
usable metric for data visualization. The end product is an Excel data file containing the cleaned dataset and relevant
cross-tabulations and a PDF(?) containing eploratory data visualizations.
Data Source: https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data

Date of Data Download: September 5, 2025

Key Limitations:

Results: This code file produces an Excel file with multiple tabs containing cleaned data and cross tabulations, as well as
three visuals: a line plot showing median medicaid enrollment in Washington D.C. from 2017 to 2025;
a dumbbell plot showing the total medicaid enrollment across mid atlantic states in December 2019 versus December 2024;
and a line plot showing median CHIP and Medicaid enrollment for mid atlantic states from 2013 to 2025.
'''

# Import relevant packages needed for data exploration and analysis
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import textwrap
import logging
import os
from datetime import datetime

# -------------------------------------------------------
# LOGGING SETUP
# Log file is saved to the same directory as my outputs.
# Each run creates a timestamped entry so logs accumulate
# across runs rather than overwriting each other.
# -------------------------------------------------------
log_dir = '/Users/emilyquick-cole/Documents/Python/medicaid_data'
log_path = os.path.join(log_dir, 'medicaid_analyzer.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),  # writes to log file
        logging.StreamHandler()  # also prints to console
    ]
)

logger = logging.getLogger(__name__)
logger.info('=' * 60)
logger.info('Medicaid Data Analyzer — Run started')
logger.info('=' * 60)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
data_path = '/Users/emilyquick-cole/Downloads/MedicaidChipApps.EligibilityDet.Enroll.9.6.2025.csv'
# Print the datapath name to the log
logger.info(f'Loading data from: {data_path}')

df = pd.read_csv(data_path)
logger.info(f'Data loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns')

# -------------------------------------------------------
# DATA EXPLORATION
# -------------------------------------------------------
logger.info('--- Data Exploration ---')

print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())

missing = df.isnull().sum()
logger.info(f'Missing value check complete — columns with missing values:\n{missing[missing > 0]}')

# -------------------------------------------------------
# DROP FOOTNOTE COLUMNS
# -------------------------------------------------------
logger.info('--- Dropping footnote columns ---')

fncol = [col for col in df.columns if col.endswith('- footnotes')]
# Prints out the names of the columns that were tagged as ending in footnote and dropped
logger.info(f'Footnote columns identified for removal: {fncol}')

# Sets new dataframe as one without the footnote columns
df = df.drop(columns=fncol, axis=1)

# Print the remaining number of columns to the log
logger.info(f'Footnote columns dropped — {df.shape[1]} columns remaining')

print(df.columns)

# -------------------------------------------------------
# FILTER TO UPDATED RECORDS ONLY
# -------------------------------------------------------
logger.info('--- Filtering to Updated records ---')

#cast the Preliminary or Updated column to a string
df['Preliminary or Updated'] = df['Preliminary or Updated'].astype('string')
#Record in the log that we did this
logger.info(f'"Preliminary or Updated" column converted to string dtype')

print(df['Preliminary or Updated'])

#Filter the dataframe to only rows where the Preliminary or Updated column contains a 'U'
updated_df = df[df['Preliminary or Updated'] == 'U']
#save the number of rows that were excluded after filtering
excluded = len(df) - len(updated_df)
# Record to the log the number of rows that we kept and excluded
logger.info(f'Filter applied — {len(updated_df)} Updated records retained, {excluded} non-Updated records excluded')

print(updated_df['Preliminary or Updated'])

# sum and save the number of duplicates
dupes = updated_df.duplicated().sum()

# Record whether duplicate records were found in the record
if dupes > 0:
    logger.warning(f'Duplicate rows detected: {dupes}')
else:
    logger.info(f'No duplicate rows found')
#Print the updated_df to an Excel file for manual checks, as needed
updated_df.to_excel('/Users/emilyquick-cole/Documents/Python/medicaid_data/updated_dataframe.xlsx', index=False)
# Record printing the updated_df to an Excel for manual checks in the log file
logger.info('Updated dataframe exported to updated_dataframe.xlsx')

# -------------------------------------------------------
# SORT DATA
# -------------------------------------------------------
logger.info('--- Sorting data by State Name and Reporting Period ---')

#make a new dataframe where the State Name and reporting period are sorted
sorted_df = (updated_df.sort_values(by=['State Name', 'Reporting Period']))
#Record this step in the log file
logger.info('Data sorted successfully by State Name and Reporting Period.')

#Export the sorted_df to Excel for manual checks
sorted_df.to_excel('/Users/emilyquick-cole/Documents/Python/medicaid_data/sorted_dataframe.xlsx', index=False)
logger.info('Sorted dataframe exported to sorted_dataframe.xlsx')

# -------------------------------------------------------
# PARSE REPORTING PERIOD INTO YEAR AND MONTH
# -------------------------------------------------------
logger.info('--- Parsing Reporting Period into Fiscal Year and Month columns ---')
#insert two new columns to represent the fiscal year and month
sorted_df.insert(3, "Reporting Period Fiscal Year", value=np.nan)
sorted_df.insert(4, "Reporting Period Month", value=np.nan)

#convert teh reporting period column to a string type
sorted_df['Reporting Period'] = sorted_df['Reporting Period'].astype('string')

#Fill the new columns by slicing the reporting period
sorted_df['Reporting Period Fiscal Year'] = sorted_df['Reporting Period'].str.slice(0, 4)
sorted_df['Reporting Period Month'] = sorted_df['Reporting Period'].str.slice(4, 6)

#Print the range of the fiscal years to the log
logger.info(
    f'Fiscal Year range: {sorted_df["Reporting Period Fiscal Year"].min()} to {sorted_df["Reporting Period Fiscal Year"].max()}')
#Print the list of all the months present in the reporting period month column to the log
logger.info(
    f'Months present: {sorted_df["Reporting Period Month"].unique().tolist() if False else sorted_df["Reporting Period Month"].unique().tolist()}')


print(sorted_df['Reporting Period Fiscal Year'])
print(sorted_df['Reporting Period Month'])

# -------------------------------------------------------
# FREQUENCY TABLES
# -------------------------------------------------------
logger.info('--- Generating frequency tables ---')

#cast the state name and reporting period fiscal year  column as categorical
sorted_df['State Name'] = sorted_df['State Name'].astype('category')
sorted_df['Reporting Period Fiscal Years'] = sorted_df['Reporting Period Fiscal Year'].astype('category')

#create frequency table of state name and reporting period fiscal years and months
state_fy_freqtable = pd.crosstab(sorted_df['State Name'], sorted_df['Reporting Period Fiscal Years'])
state_month_freqtable = pd.crosstab(sorted_df['State Name'], sorted_df['Reporting Period Month'])

#Record the rows and columns of the state by month frequency table and the table in the log file
logger.info(
    f'State by FY frequency table generated — {len(state_fy_freqtable)} states across {len(state_fy_freqtable.columns)} fiscal years')
#logger.info(f"\n--- Cross-Tabulation Log ---\n{state_fy_freqtable.to_string()}\n----------------------------")

#Record the rows and columns of the state by month frequency table and the table in the log file
logger.info(f'State by Month frequency table generaged - {len(state_month_freqtable)} states across {len(state_month_freqtable.columns)} months.')
#logger.info(f"\n--- Cross-Tabulation Log ---\n{state_month_freqtable.to_string()}\n----------------------------")


print(state_fy_freqtable)
print(state_month_freqtable)

#export the sorted dataframe and both frequency tables to an Excel file
with pd.ExcelWriter('/Users/emilyquick-cole/Documents/Python/medicaid_data/sorted_dataframe.xlsx',
                    engine='openpyxl') as writer:
    sorted_df.to_excel(writer, sheet_name="Sorted Data", index=False)
    state_fy_freqtable.to_excel(writer, sheet_name="Reporting Freq FY", index=True)
    state_month_freqtable.to_excel(writer, sheet_name="Reporting FY Month", index=True)

#Record exporting the sorted dataframe and both frequency tables to an Excel file
logger.info('Sorted dataframe and frequency tables exported to sorted_dataframe.xlsx')

# -------------------------------------------------------
# VISUALIZATION 1: DC Medicaid and CHIP Enrollment Line Plot
# -------------------------------------------------------
logger.info('--- Figure 1: DC Medicaid and CHIP enrollment line plot ---')

# Cast the State Name to a string
sorted_df['State Name'] = sorted_df['State Name'].astype('string')

#filter the dataframe to rows that have D.C. as the State
dc_df = sorted_df[sorted_df['State Name'] == 'District of Columbia']

#Find the median Medicaid Enrollment by Reporting Period Fiscal Year
median_by_fy = dc_df.groupby('Reporting Period Fiscal Year')['Total Medicaid Enrollment'].median()
#Find the median Medicaid CHIP Enrollment by Reporting Period Fiscal Year
chip_by_fy = dc_df.groupby('Reporting Period Fiscal Year')['Total CHIP Enrollment'].median()

#Flag if the state D.C is not in the sorted_dataframe
if 'District of Columbia' not in sorted_df['State Name'].values:
    logger.warning('District of Columbia not found in dataset — Figure 1 may be empty or incorrect')
else:
    logger.info('District of Columbia records confirmed present for Figure 1')


dc_med_enroll = pd.DataFrame(median_by_fy)
dc_chip_enroll = pd.DataFrame(chip_by_fy)
dc_enroll = pd.concat([dc_med_enroll, dc_chip_enroll], axis=1)
dc_enroll = dc_enroll.reset_index()

#logger.info(f"\n--- Data for Visual 1 ---\n{dc_enroll.to_string()}\n----------------------------")

f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

#Create the line for Total Medicaid Enrollment
ax1 = sns.lineplot(data=dc_enroll, x='Reporting Period Fiscal Year',
                   y='Total Medicaid Enrollment', ax=ax1,
                   color='blue', marker="o", label='Medicaid')
# Create the line for Total CHIP Enrollment
ax2 = sns.lineplot(data=dc_enroll, x='Reporting Period Fiscal Year',
                   y='Total CHIP Enrollment', ax=ax2,
                   color='orange', marker='o', label='CHIP')
#Format the line plot
ax1.set_ylim(800000, 1200000)
ax2.set_ylim(40000, 100000)
ax1.ticklabel_format(style='plain', useOffset=False, axis='y')
ax2.ticklabel_format(style='plain', useOffset=False, axis='y')
ax1.get_xaxis().set_visible(False)

f.text(0.08, 0.93, "Median Medicaid Enrollment in D.C. From 2017 to 2025",
       va="center", rotation="horizontal", fontsize=16, weight="bold")
f.text(x=0.45, y=0.07, s="Fiscal Year", va="center",
       rotation="horizontal", fontsize=12, weight="bold")
plt.xlabel("")
ax1.set_ylabel("")
ax2.set_ylabel("")
f.text(0.03, 0.55, "Total Enrolled", va="center",
       rotation="vertical", fontsize=12, weight="bold")
ax1.xaxis.tick_top()
ax2.xaxis.tick_bottom()
f.subplots_adjust(left=0.3, right=0.97, bottom=0.15, top=0.85)

#Export the figure
fig1_path = '/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig1.png'
plt.savefig(fig1_path)
plt.close()

#Record the file path
logger.info(f'Figure 1 saved to {fig1_path}')

# -------------------------------------------------------
# VISUALIZATION 2: Mid-Atlantic Dumbbell Chart
# -------------------------------------------------------
logger.info('--- Figure 2: Mid-Atlantic dumbbell chart ---')

#Create a list of states that are mid-atlantic
mid_atl = ['New York', 'New Jersey', 'Pennsylvania', 'Delaware', 'Maryland', 'Virginia', 'West Virginia',
           'District of Columbia']
#select the columns from sorted_df that we want to produce the next visual
dbell_data = sorted_df[
    ['State Name', 'Reporting Period Fiscal Year', 'Reporting Period Month', 'Total Medicaid Enrollment']]
#filter the dataframe to only rows where the fiscal year is 2019 or 2024
dbell_data = dbell_data[
    (dbell_data['Reporting Period Fiscal Year'] == '2019') | (dbell_data['Reporting Period Fiscal Year'] == '2024')]

#filter the data so that we only include data from December
dbell_data = dbell_data[(dbell_data['Reporting Period Month'] == '12')]

#filter the data to only include states that are mid-atlantic states
dbell_data = dbell_data[(dbell_data['State Name'].isin(mid_atl))]

#save value counts for each state in the data frame
state_counts = dbell_data['State Name'].value_counts()
#Log the number of states and the total records -- should be 16.
logger.info(f'Dumbbell chart data — {len(state_counts)} states, {sum(state_counts)} total records')

# Create a flag in the log if the number of datapoints in the filtered data is incorrect.
if sum(state_counts) == 16:
    logger.info(f'There are 8 states and 16 datapoints in the filtered data, which is what is expected.')
else:
    logger.info(f'There is an incorrect number of datapoints in the filtered data.')

# Save any states in mid-atlantic that do not appear in the datasets
missing_states = [s for s in mid_atl if s not in state_counts.index]
# If there are any missing states, print these to the log
if missing_states:
    logger.warning(f'The following Mid-Atlantic states are missing from dumbbell data: {missing_states}')

# If there is a state that has less than 2 points, print these to the log
states_with_one_period = state_counts[state_counts < 2].index.tolist()
if states_with_one_period:
    logger.warning(f'These states are missing data for one of the two time periods: {states_with_one_period}')

print("There is data for ", len(state_counts), " states and a total of ", sum(state_counts), "records.")

#Print the data for visual 2 to the log
#logger.info(f"\n--- Data for Visual 2 ---\n{dbell_data.to_string()}\n----------------------------")

#Set the state name and reporting period year column as a categorical type
dbell_data['State Name'] = dbell_data['State Name'].astype('category')
dbell_data['Reporting Period Fiscal Year'] = dbell_data['Reporting Period Fiscal Year'].astype('category')
#reset the index of the data set
dbell_data = dbell_data.reset_index()

#Format the visual
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))

sns.pointplot(data=dbell_data, x='Total Medicaid Enrollment', y='State Name', hue='Reporting Period Fiscal Year',
              dodge=False, join=False, marker='o', markersize=10, ax=ax)
#pivot the dbell dataframe
pivot_dbell = dbell_data.pivot(index='State Name', columns='Reporting Period Fiscal Year',
                               values='Total Medicaid Enrollment').reset_index()
#iterate over the pivoted dataframe to plot the datapoints
for index, row in pivot_dbell.iterrows():
    ax.plot([row['2019'], row['2024']], [row['State Name'], row['State Name']], color='gray', linestyle='-',
            linewidth=1)

ax.set_title('Difference in Total Medicaid Enrollment in Mid-Atlantic States From December 2019 to December 2024',
             fontsize=14, weight='bold')
ax.set_xlabel('People Enrolled', fontsize=14, weight='bold')
ax.set_ylabel('')


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)


wrap_labels(ax, 13)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['2019', '2024'], title='Fiscal Year')
ax.ticklabel_format(useOffset=False, style='plain', axis='x')
ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend(bbox_to_anchor=(1.01, 0.55), loc='upper left', borderaxespad=0)

fig2_path = '/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig2.png'
plt.savefig(fig2_path)
plt.close()

#Save the file path for visual 2 to the log
logger.info(f'Figure 2 saved to {fig2_path}')

# -------------------------------------------------------
# VISUALIZATION 3: Mid-Atlantic CHIP and Medicaid Line Plot
# -------------------------------------------------------
logger.info('--- Figure 3: Mid-Atlantic CHIP and Medicaid line plot ---')

#group the sorted dataframe by State Name and Reporting Period Fiscal Year and take the median of Total Medicaid and CHIP Enrollment
state_med_by_year = sorted_df.groupby(['State Name', 'Reporting Period Fiscal Year'])[
    'Total Medicaid and CHIP Enrollment'].median()
#reset the index of this grouped dataframe
state_med_by_year = pd.DataFrame(state_med_by_year).reset_index()
#filter the dataframe to only include rows with State Name in mid-atl states
midatl_med_by_year = state_med_by_year[(state_med_by_year['State Name'].isin(mid_atl))]

# Print the unique states and the number of records in this filtered dataframe
logger.info(
    f'Figure 3 data — {midatl_med_by_year["State Name"].nunique()} Mid-Atlantic states, {len(midatl_med_by_year)} records')

#Print the data for visual 3 to the log
#logger.info(f"\n--- Data for Visual 3 ---\n{midatl_med_by_year.to_string()}\n----------------------------")

#Set up the lineplot
g = sns.lineplot(data=midatl_med_by_year, x="Reporting Period Fiscal Year",
                 y="Total Medicaid and CHIP Enrollment", hue="State Name",
                 palette="flare", style='State Name')

sns.move_legend(ax, "lower center", bbox_to_anchor=(1, 1))
sns.set_theme(style="darkgrid")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.75))

g.set_title("Median CHIP and Medicaid Enrollment for Mid-Atlantic States from 2013 to 2024", fontsize=12, weight='bold')
g.set_xlabel("Fiscal Year", fontsize=12, weight='bold')
g.set_ylabel('People Enrolled', fontsize=12, weight='bold')
plt.setp(g.get_legend().get_title(), weight='bold')
plt.ticklabel_format(style='plain', axis='y')
g.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

fig3_path = '/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig3.png'
plt.savefig(fig3_path, bbox_inches="tight")
plt.close()

#Print the filepath to figure 3 to the log
logger.info(f'Figure 3 saved to {fig3_path}')

# -------------------------------------------------------
# RUN COMPLETE
# -------------------------------------------------------
logger.info('=' * 60)
logger.info('Medicaid Data Analyzer — Run complete')
logger.info('=' * 60)
