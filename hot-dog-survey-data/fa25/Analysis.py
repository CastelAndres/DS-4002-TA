import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#Load the Excel file
df = pd.read_excel("Hot_Dog_Data.xlsx")
#Remove the Timestamp column
df = df.drop("Timestamp", axis=1)
#Renaming columns to make it easier 
df = df.rename(columns={ "What year are you in?": "Year", "Is hotdog a sandwich?": "Sandwich?"})
# Preview the first rows
print(df.head())
#Below we begin to start some exploratory plots
import matplotlib.pyplot as plt
import seaborn as sns

#Histogram of yes/no by year
sns.histplot(data=df, x="Year", hue="Sandwich?", multiple="dodge", shrink=0.8)
plt.title("Is a Hotdog sandwich? (by Year)")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

#Overall countplot of Yes/No from all Undergraduates
sns.countplot(x="Sandwich?", data=df, palette="Set2")
plt.title("Overall Responses to 'Is a Hot Dog a Sandwich?'")
plt.xlabel("Response")
plt.ylabel("Count")
plt.show()

#Checking for any Bias in the data
#Looked at proportion + 95% CI for each year
df["Yes_binary"] = df["Sandwich?"].map({"Yes": 1, "No": 0})
prop_by_year = df.groupby("Year")["Yes_binary"].agg(['mean','count']).reset_index()
prop_by_year["se"] = (prop_by_year["mean"] * (1 - prop_by_year["mean"]) / prop_by_year["count"])**0.5

#Below is the 95% confidence interval
prop_by_year["lower"] = prop_by_year["mean"] - 1.96 * prop_by_year["se"]
prop_by_year["upper"] = prop_by_year["mean"] + 1.96 * prop_by_year["se"]
print(prop_by_year)

#Below we plot it
plt.errorbar(prop_by_year["Year"], prop_by_year["mean"], 
             yerr=1.96*prop_by_year["se"], fmt="o", capsize=4)
plt.title("Proportion Saying 'Yes' with 95% CI")
plt.ylabel("Proportion Yes")
plt.ylim(0,1)
plt.show()

#Below we bootstramp sample to check for uncertainty
def bootstrap_proportion(data, n_boot=1000):
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(sample.mean())
    return np.percentile(boot_means, [2.5, 97.5])
#Example for one year
year_data = df.loc[df["Year"] == 2020, "Yes_binary"].values
ci = bootstrap_proportion(year_data)
print("Bootstrap 95% CI for 2020:", ci)