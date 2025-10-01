
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

#Load your dataset (e.g., CSV file). #Edit pathname to where file is located on your device.
df = pd.read_csv('/Users/emilyquick-cole/Downloads/MedicaidChipApps.EligibilityDet.Enroll.9.6.2025.csv')

'''Data Exploration'''

#print the titles of the column headers
print(df.columns)
#print the column types
print(df.dtypes)
#print info about the data set
print(df.info())
print(df.describe()) #gives descriptive stats of numerical columns
#Check for missing values
print("Missing Values", df.isnull().sum())


'''Drop any columns with "footnote" in the title '''

#Save columns that end with "footnote" to a variable
fncol = [col for col in df.columns if col.endswith('- footnotes')]

#Drop the columns in this variable from the dataframe
df = df.drop(columns=fncol, axis=1)

#Check that the columns were dropped.
print(df.columns)

'''Filter data so that it only contains 'Updated' Records '''
#The Preliminary or Updated column indicates the status of a record. Filter to only include the Updated Columns

#First convert the column value to a string
df['Preliminary or Updated'] = df['Preliminary or Updated'].astype('string')
#Check that the column was successfully converted to a string type
print(df['Preliminary or Updated'].dtype)

#Filter the data to only include "Updated" records
print(df['Preliminary or Updated'])
#Create a new data frame that only contains updated records
updated_df = df[df['Preliminary or Updated'] == 'U']
#Check that the Preliminary or Updated column in the Updated Dataframe contains only records with "U"
print(updated_df['Preliminary or Updated'])
# Check for duplicate rows, there are 0
print("Total number of duplicated rows", updated_df.duplicated().sum())

#Output the current dataset of updated data to an Excel file to review the information. 
#Edit pathname to where you want file to be located on your device.
updated_df.to_excel('/Users/emilyquick-cole/Documents/Python/medicaid_data/updated_dataframe.xlsx', index=False)

'''Comments Based on Review of Excel
The reporting period identifier indicates the year and month that the data pertains to. These data, combined with the 
state identifier, can be combined to create a unique identifier. The reporting period column can also be split to create
a fiscal year and month column. 

Key data columns we're interested in: Total Medicaid Enrollment, Total CHIP Enrollment
Compare stats for expanded vs non-expanded medicaid? 
'''

#Order the rows alphabetically by state and within that, by reporting period
sorted_df = (updated_df.sort_values(by = ['State Name', 'Reporting Period']))

#Output the current dataset of updated data to an Excel file to review the information
#Edit pathname to where you want file to be located on your device.
sorted_df.to_excel('/Users/emilyquick-cole/Documents/Python/medicaid_data/sorted_dataframe.xlsx', index=False)

''' Decipher the Reporting Period Identifier for each row'''
# Make new, empty columns.
sorted_df.insert(3, "Reporting Period Fiscal Year", value = np.nan)
sorted_df.insert(4, "Reporting Period Month", value = np.nan)

#Check that the columns were added successfully
print(sorted_df.columns)

#Iterate through Reporting Period identifier and populate data into the other columns

#set Reporting Period as a string variable
sorted_df['Reporting Period'] = sorted_df['Reporting Period'].astype('string')
#Pull out the first four characters of the reporting period identifier and save as fiscal year. Pull out the
#last two characters of reporting period identifier and save as month
for row in sorted_df['Reporting Period']:
    sorted_df['Reporting Period Fiscal Year'] = sorted_df['Reporting Period'].str.slice(0, 4)
    sorted_df['Reporting Period Month'] = sorted_df['Reporting Period'].str.slice(4, 6)
    break
#Check that the values were saved appropriately
print(sorted_df['Reporting Period Fiscal Year'])
print(sorted_df['Reporting Period Month'])

'''How many records were submitted by each state across all fiscal years, and for how many months within each fiscal year?'''

#To answer this question, first set our two variables of interest as categorical
sorted_df['State Name'] = sorted_df['State Name'].astype('category')
sorted_df['Reporting Period Fiscal Years'] = sorted_df['Reporting Period Fiscal Year'].astype('category')

#Use the cross tab function to compare how many states reported in each fiscal year, and how many times
state_fy_freqtable = pd.crosstab(sorted_df['State Name'],sorted_df['Reporting Period Fiscal Years'])
print(state_fy_freqtable)

#Use the same function above for states and reporting months
state_month_freqtable = pd.crosstab(sorted_df['State Name'],sorted_df['Reporting Period Month'])
print(state_month_freqtable)

#Output an Excel file that contains the complete sorted dataframe, the State by reporting fiscal year frequency table,
#and the state by reporting month frequency table

#Set the index as true so that the state labels are shown
#Edit pathname to where you want file to be located on your device.
with pd.ExcelWriter('/Users/emilyquick-cole/Documents/Python/medicaid_data/sorted_dataframe.xlsx', engine='openpyxl') as writer:
    sorted_df.to_excel(writer,sheet_name="Sorted Data", index=False)
    state_fy_freqtable.to_excel(writer, sheet_name="Reporting Freq FY", index=True)
    state_month_freqtable.to_excel(writer, sheet_name="Reporting FY Month", index=True)

'''Visualizations'''

'''Make a line graph showing the fluctuations in median total enrollment over time for both Medicaid and CHIP'''

#set State Name to a categorial variable
sorted_df['State Name'] = sorted_df['State Name'].astype('string')

#iterate through and take the median total medicaid enrollment for each fiscal year, only for Washington DC
for i in sorted_df['State Name']:
    if i == 'District of Columbia':
        median_by_fy = sorted_df.groupby('Reporting Period Fiscal Year')['Total Medicaid Enrollment'].median()
        chip_by_fy = sorted_df.groupby('Reporting Period Fiscal Year')['Total CHIP Enrollment'].median()
        print("Median Medicaid by FY", median_by_fy)
        print("Median CHIP by FY", chip_by_fy)

#Convert both outputs to a data frame
dc_med_enroll = pd.DataFrame(median_by_fy)
dc_chip_enroll = pd.DataFrame(chip_by_fy)

#Concatonate the two dataframes
dc_enroll= pd.concat([dc_med_enroll, dc_chip_enroll], axis=1)
print(dc_enroll.columns)

#Reset the index so that the fiscal years are interpreted as a column, not an index.
dc_enroll = dc_enroll.reset_index()
#Check that the index reset
print("The columns are", dc_enroll.columns)

#We want the graphic to have a broken y-axis to the differences in enrollment between Medicaid and CHIP

#This is the figure our two plots will reside on we need a lower part (anything below the cutoff), which will be ax2
#and an upper part (anything above the cutoff) which will be ax1
#because we have only two plots above each other, we set ncols=1 and nrows=2
#also, they should share an x axis, which is why we set sharex=True
f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

#Plot the two graphs
ax1 = sns.lineplot(data = dc_enroll, x= 'Reporting Period Fiscal Year',
                   y='Total Medicaid Enrollment', ax=ax1,
                   color = 'blue', marker = "o", label = 'Medicaid') #color = 'blue'
ax2 = sns.lineplot(data = dc_enroll, x = 'Reporting Period Fiscal Year',
                   y = 'Total CHIP Enrollment', ax = ax2,
                   color = 'orange', marker = 'o', label = 'CHIP', ) #color = 'orange'


#Set the limits for each individual y-axis
ax1.set_ylim(800000, 1200000)
ax2.set_ylim(40000, 100000)

#Remove scientific notation from both y-axes
ax1.ticklabel_format(style='plain', useOffset=False, axis='y')
ax2.ticklabel_format(style='plain', useOffset=False, axis='y')

#The upper part does not need its own x-axis as it shares one with the lower part
ax1.get_xaxis().set_visible(False)

#Create a title label and label for the title and x-axis
f.text(0.08, 0.93, "Median Medicaid Enrollment in D.C. From 2017 to 2025",
       va="center", rotation="horizontal", fontsize=16, weight = "bold")
f.text(x = 0.45, y = 0.07, s="Fiscal Year", va="center",
       rotation="horizontal", fontsize=12, weight = "bold")
plt.xlabel("")

#By default, each part will get its own "Total enrolled" y-axis label, but we want to set a common for the whole figure
#Remove the y label for both subplots
ax1.set_ylabel("")
ax2.set_ylabel("")
# then, set a new label on the plot (basically just a piece of text) and move it to where it makes sense (requires trial and error)
f.text(0.03, 0.55, "Total Enrolled", va="center",
       rotation="vertical", fontsize = 12, weight = "bold")

#Put some ticks on the top of the upper part and bottom of the lower part for style
ax1.xaxis.tick_top()
ax2.xaxis.tick_bottom()

#Finally, adjust everything a bit to make it prettier (this just moves everything, best to try and iterate)
f.subplots_adjust(left=0.3, right=0.97, bottom=0.15, top=0.85)

# Saving the plot as a PNG file
#Edit pathname to where you want file to be located on your device.
plt.savefig("/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig1.png")
# add plt.close() after you've saved the figure
plt.close()

'''Compare the Total Medicaid nrollment values for Mid Atlantic States at end of 2019 to end of 2024
A dumbbell chart can take into account that we have 50 states + DC, two distinct points in time, and a numerical
variable of Medicaid and CHIP enrollment
'''
'''Format the data to make the dummbbell plot '''
dbell_data= sorted_df[['State Name', 'Reporting Period Fiscal Year', 'Reporting Period Month', 'Total Medicaid Enrollment']]
#Filter to only include records from 2019 and 2024
dbell_data = dbell_data[(dbell_data['Reporting Period Fiscal Year'] == '2019') | (dbell_data['Reporting Period Fiscal Year'] == '2024')]

#Filter data to only include December records
dbell_data = dbell_data[(dbell_data['Reporting Period Month'] == '12')]

#Filter data to only include Mid Atlantic State Records
# Includes New York, New Jersey, Pennsylvania, Delaware, Maryland, Virginia, West Virginia, and the national capital of Washington, D.C.
mid_atl = ['New York', 'New Jersey', 'Pennsylvania', 'Delaware', 'Maryland', 'Virginia', 'West Virginia', 'District of Columbia']
dbell_data = dbell_data[(dbell_data['State Name'].isin(mid_atl))]

#Confirm that each state has two records
state_counts = dbell_data['State Name'].value_counts()
print("There is data for ", len(state_counts), " states and a total of ", sum(state_counts), "records.")

#set the State and Reporting period categories as categorical
dbell_data['State Name'] = dbell_data['State Name'].astype('category')
dbell_data['Reporting Period Fiscal Year'] = dbell_data['Reporting Period Fiscal Year'].astype('category')
#Reset the index
dbell_data = dbell_data.reset_index()

'''Develop the visual'''
# Set up the plot style
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the markers using seaborn's pointplot
# `dodge` is set to False so the markers are plotted on the same vertical line
sns.pointplot(data=dbell_data, x='Total Medicaid Enrollment', y='State Name', hue='Reporting Period Fiscal Year',
              dodge=False, join=False, marker='o',
              markersize=10, ax=ax)

# Draw the connecting lines manually using matplotlib
pivot_dbell = dbell_data.pivot(index = 'State Name',
                               columns = 'Reporting Period Fiscal Year',
                               values = 'Total Medicaid Enrollment').reset_index()

#iterate through rows to draw the lines connecting points
for index, row in pivot_dbell.iterrows():
    #ax.plot([row['2019'], row['2024']],[row['State Name'], row['State Name']],color='gray', zorder=0)
    ax.plot([row['2019'], row['2024']], [row['State Name'], row['State Name']], color='gray', linestyle='-',
            linewidth=1)

# Add titles and labels for clarity
ax.set_title('Difference in Total Medicaid Enrollment in Mid-Atlantic States From December 2019 to December 2024', fontsize=14, weight = 'bold')
ax.set_xlabel('People Enrolled', fontsize=14, weight = 'bold')
#ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_ylabel('')  # Y-axis labels are self-explanatory

#Adjust the y labels so that they don't run off the page
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)

#wrap the labels so they all fit
wrap_labels(ax, 13)

# Improve the legend title and placement
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['2019', '2024'], title='Fiscal Year')

# Disable scientific notation and offset on the x-axis
ax.ticklabel_format(useOffset=False, style='plain', axis='x')
#Add commas to the x-axis labels
ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# Shift the plot to the right by adjusting the 'left' parameter
plt.legend(bbox_to_anchor=(1.01, 0.55), loc='upper left', borderaxespad=0)

# Saving the plot as a PNG file
#Edit pathname to where you want file to be located on your device.
plt.savefig("/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig2.png")
# add plt.close() after you've saved the figure
plt.close()

'''Final Visual: Median Enrollment of CHIP and Medicaid Enrollment for Mid-Atlantic States'''

#Use our sorted dataframe to determine the median Total CHIP and Medicaid enrollment from 2013 to 2024
state_med_by_year = sorted_df.groupby(['State Name', 'Reporting Period Fiscal Year'])['Total Medicaid and CHIP Enrollment'].median()
#print('state med by year', state_med_by_year)

#convert to a dataframe
state_med_by_year = pd.DataFrame(state_med_by_year)
state_med_by_year = state_med_by_year.reset_index()

#Filter to only include mid-atlantic states
midatl_med_by_year = state_med_by_year[(state_med_by_year['State Name'].isin(mid_atl))]

#Make the line graph
g = sns.lineplot(data = midatl_med_by_year,
             x="Reporting Period Fiscal Year",
             y="Total Medicaid and CHIP Enrollment",
             hue="State Name",
             #legend = False,
             palette = "flare",
             style = 'State Name')

# Move the legend outside the plot area
sns.move_legend(ax, "lower center", bbox_to_anchor=(1, 1))

#set the tehme
sns.set_theme(style="darkgrid")

#move the legend so it's off the grid
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.75))

#Adjust the title, axis, and legend labels
g.set_title("Median CHIP and Medicaid Enrollment for Mid-Atlantic States from 2013 to 2024", fontsize = 12, weight = 'bold')
#g.set(xlabel='Fiscal Year', ylabel='People Enrolled')
g.set_xlabel( "Fiscal Year", fontsize=12, weight = 'bold')
g.set_ylabel('People Enrolled', fontsize = 12, weight = 'bold')
plt.setp(g.get_legend().get_title(), weight = 'bold')

# Disable scientific notation and offset on the x-axis
plt.ticklabel_format(style='plain', axis='y')

#Add commas to the y-axis labels
g.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# Saving the plot as a PNG file
#Edit pathname to where you want file to be located on your device.
plt.savefig("/Users/emilyquick-cole/Documents/Python/medicaid_data/Fig3.png", bbox_inches="tight")
# add plt.close() after you've saved the figure
plt.close()

