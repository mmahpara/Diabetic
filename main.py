
# ==================================================
# DIABETES DATA ANALYSIS PROJECT
# Using Python, Pandas and Matplotlib
# ==================================================

# STEP 1: Import libraries

import pandas as pd
import matplotlib.pyplot as plt

# STEP 2: Import dataset

df = pd.read_csv(
    "diabetic_data.csv",
    na_values=["?"],
    low_memory=False
)

# STEP 3: Display first five rows

print("\nFIRST FIVE ROWS")
print(df.head())

# STEP 4: Understand the dataset

print("\nDATASET INFORMATION")
print(df.info())

# STEP 5: Check dataset shape

print("\nDATASET SHAPE")
print(df.shape)

# STEP 6: Check column names

print("\nCOLUMN NAMES")
print(df.columns.tolist())

# STEP 7: Check missing values

print("\nMISSING VALUES")
print(df.isnull().sum())

# STEP 8: Statistical summary

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# STEP 9: Set age group order

age_order = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)',
    '[40-50)', '[50-60)', '[60-70)', '[70-80)',
    '[80-90)', '[90-100)'
]

df["age"] = pd.Categorical(
    df["age"],
    categories=age_order,
    ordered=True
)

# ==================================================
# QUESTION 1:
# Which age group has the highest number of
# hospital encounters?
# ==================================================

print("\nQUESTION 1")

age_counts = df["age"].value_counts(
    sort=False
)

age_counts = age_counts[age_counts > 0]

print(age_counts)

print(
    "Highest age group:",
    age_counts.idxmax()
)

age_counts.plot(
    kind="bar",
    title="Hospital Encounters by Age Group",
    xlabel="Age Group",
    ylabel="Number of Encounters"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 2:
# What is the distribution of patient readmissions?
# ==================================================

print("\nQUESTION 2")

readmission_counts = df["readmitted"].value_counts()

readmission_order = ["NO", ">30", "<30"]

readmission_counts = readmission_counts.reindex(
    readmission_order
)

print(readmission_counts)

readmission_counts.plot(
    kind="bar",
    title="Distribution of Readmissions",
    xlabel="Readmission Status",
    ylabel="Number of Encounters"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 3:
# Which age group has the highest readmission
# percentage?
# ==================================================

print("\nQUESTION 3")

df["any_readmission"] = df["readmitted"] != "NO"

readmission_by_age = (
    df.groupby("age", observed=False)["any_readmission"]
    .mean() * 100
).dropna()

print(readmission_by_age.round(2))

print(
    "Highest readmission percentage:",
    readmission_by_age.idxmax()
)

readmission_by_age.plot(
    kind="bar",
    title="Readmission Percentage by Age Group",
    xlabel="Age Group",
    ylabel="Readmission Percentage (%)"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 4:
# What is the average hospital stay for each
# readmission category?
# ==================================================

print("\nQUESTION 4")

average_stay = (
    df.groupby("readmitted")["time_in_hospital"]
    .mean()
    .reindex(readmission_order)
)

print(average_stay.round(2))

average_stay.plot(
    kind="bar",
    title="Average Hospital Stay by Readmission Status",
    xlabel="Readmission Status",
    ylabel="Average Days in Hospital"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 5:
# Which diabetes medication is most commonly
# recorded as prescribed?
# ==================================================

print("\nQUESTION 5")

medication_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

medication_counts = (
    df[medication_columns]
    .isin(["Steady", "Up", "Down"])
    .sum()
    .sort_values(ascending=False)
)

print(medication_counts)

print(
    "Most commonly recorded medication:",
    medication_counts.idxmax()
)

medication_counts.head(10).sort_values().plot(
    kind="barh",
    title="Top 10 Recorded Diabetes Medications",
    xlabel="Number of Encounters",
    ylabel="Medication"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 6:
# How is insulin treatment status associated
# with readmission?
# ==================================================

print("\nQUESTION 6")

insulin_readmission = pd.crosstab(
    df["insulin"],
    df["readmitted"],
    normalize="index"
) * 100

insulin_readmission = insulin_readmission.reindex(
    index=["No", "Down", "Steady", "Up"],
    columns=readmission_order
)

print(insulin_readmission.round(2))

insulin_readmission.plot(
    kind="bar",
    stacked=True,
    title="Insulin Status and Readmission",
    xlabel="Insulin Treatment Status",
    ylabel="Percentage (%)"
)

plt.legend(title="Readmission Status")

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 7:
# Does the number of previous inpatient visits
# relate to readmission rates?
# ==================================================

print("\nQUESTION 7")

df["previous_inpatient_group"] = pd.cut(
    df["number_inpatient"],
    bins=[-1, 0, 1, 2, float("inf")],
    labels=["0 visits", "1 visit", "2 visits", "3+ visits"]
)

previous_readmission = (
    df.groupby(
        "previous_inpatient_group",
        observed=False
    )["any_readmission"]
    .mean() * 100
)

print(previous_readmission.round(2))

previous_readmission.plot(
    kind="bar",
    title="Previous Inpatient Visits and Readmission",
    xlabel="Previous Inpatient Visits",
    ylabel="Readmission Percentage (%)"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 8:
# How does the number of laboratory procedures
# vary with hospital stay duration?
# ==================================================

print("\nQUESTION 8")

lab_by_stay = (
    df.groupby("time_in_hospital")["num_lab_procedures"]
    .mean()
)

print(lab_by_stay.round(2))

lab_by_stay.plot(
    kind="line",
    marker="o",
    title="Laboratory Procedures by Hospital Stay",
    xlabel="Days in Hospital",
    ylabel="Average Number of Lab Procedures"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 9:
# Which age group receives the highest average
# number of medications?
# ==================================================

print("\nQUESTION 9")

medications_by_age = (
    df.groupby("age", observed=False)["num_medications"]
    .mean()
).dropna()

print(medications_by_age.round(2))

print(
    "Age group with highest average medications:",
    medications_by_age.idxmax()
)

medications_by_age.plot(
    kind="bar",
    title="Average Number of Medications by Age",
    xlabel="Age Group",
    ylabel="Average Number of Medications"
)

plt.tight_layout()
plt.show()

# ==================================================
# QUESTION 10:
# Which columns have the most missing data?
# ==================================================

print("\nQUESTION 10")

missing_data = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": df.isnull().mean() * 100
})

missing_data = missing_data[
    missing_data["Missing Values"] > 0
].sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_data.round(2))

missing_data.head(10)["Missing Percentage"].sort_values().plot(
    kind="barh",
    title="Top 10 Columns with Missing Data",
    xlabel="Missing Values (%)",
    ylabel="Column Name"
)

plt.tight_layout()
plt.show()

# ==================================================
# PROJECT COMPLETED
# ==================================================

print("\nALL 10 ANALYSES COMPLETED SUCCESSFULLY!")