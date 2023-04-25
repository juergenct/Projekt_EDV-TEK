import pandas as pd
import matplotlib.pyplot as plt

# Read in json file as Pandas dataframe
df_USPTO_Cleantech = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_Cleantech_Patents_Statistics.json')
df_USPTO_non_Cleantech = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_Non_Cleantech_Patents_Statistics.json')
# df_USPTO = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2013_2023.json')

# # Sum up the number of Cleantech patents in df_USPTO_Cleantech
# df_USPTO_Cleantech_sum = df_USPTO_Cleantech.sum(axis=0)
# df_USPTO_Non_Cleantech_sum = len(df_USPTO) - df_USPTO_Cleantech_sum[0]

# # Plot the sum of Cleantech and non Cleantech patents as pie chart
# fig, ax = plt.subplots()
# ax.pie([df_USPTO_Cleantech_sum[0], df_USPTO_Non_Cleantech_sum], labels=['Cleantech', 'Non Cleantech'], autopct='%1.1f%%', shadow=True, startangle=90)
# ax.axis('equal')
# ax.text(-0.4, 0.65, df_USPTO_Cleantech_sum[0], fontsize=10, color='black')
# ax.text(0, -0.7, df_USPTO_Non_Cleantech_sum, fontsize=10, color='black')
# plt.show()

# Plot the distribution of Cleantech patents from Pandas dataframe in a pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(df_USPTO_Cleantech['Cleantech Patents'], 
       labels=df_USPTO_Cleantech.index, 
       shadow=True, 
       startangle=90,
       autopct='%1.1f%%')  # Display values in percentages
ax.axis('equal')
plt.tight_layout()  # Check for overlapping text
plt.show()
