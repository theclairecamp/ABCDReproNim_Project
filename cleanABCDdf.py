#!/usr/bin/env python
# coding: utf-8

# First load in container for python analysis below.
# -

# Start python analysis - First collapse all txt files from the ABCD 3.0 data release into one dataframe, with no duplicates:
# -
# 
# Code below is adapted from the ABCD ReproNim Course created by James Kent https://github.com/ABCD-ReproNim/exercises/blob/main/break/viewABCD.md

# In[1]:


#Import libraries
import pandas as pd # to read/manipulate/write data from files
import numpy as np # to manipulate data/generate random numbers

from pathlib import Path # represent and interact with directories/folders in the operating system
from collections import namedtuple # structure data in an easy to consume way

import requests # retrieve data from an online source


# In[2]:


# save directory we downloaded the ABCD data to `data_path`
data_path = Path("ABCD3/")
# glob (match) all text files in the `data_path` directory
files = sorted(data_path.glob("*.txt"))


# In[3]:


#Read in data from above cell
data_elements_df = pd.read_csv('data_elements.tsv', sep='\t')


# In[4]:


#use to find whether the variable exists in our data_elements file
data_elements_df.query("element == 'mri_info_manufacturersmn'")


# In[5]:


#load in data dictionary
data_dic = pd.read_excel (r'Data_Dictionary.xls')
#check if loaded in correcly
data_dic.head()


# In[6]:


#variables needed for our data analysis from data dictionary
common = ["eventname","subjectkey","interview_age","sex"]
demographic = ["site_id_l","rel_family_id","ehi_y_ss_scoreb","demo_comb_income_v2","demo_prnt_ed_v2","race_ethnicity","demo_ed_v2","medhx_ss_9b_p","rel_relationship"]
clinical = ["cbcl_scr_syn_anxdep_r","cbcl_scr_syn_withdep_r","cbcl_scr_syn_somatic_r","cbcl_scr_syn_social_r","cbcl_scr_syn_thought_r","cbcl_scr_syn_attention_r","cbcl_scr_syn_rulebreak_r","cbcl_scr_syn_aggressive_r"]
behavioral = [] #dont have any
cognitive = [] #dont have any
imaging = data_dic['Variable Code'].tolist()[21:]

data_elements_of_interest = demographic + clinical + behavioral + cognitive + imaging


# In[7]:


print(data_dic.loc[[21]])


# In[8]:


#create dictionary whose keys are the data structures and whose values are the data elements of interest 
#within that data structure
structures2read = {}
for element in data_elements_of_interest:
    item = data_elements_df.query(f"element == '{element}'").structure.values[0]
    if item not in structures2read:
        structures2read[item] = []
    structures2read[item].append(element)


# In[9]:


#load in all data structures and combine into one df based on the variables inserted earlier
all_df = None
for structure, elements in structures2read.items():
    data_structure_filtered_df = pd.read_table(data_path / f"{structure}.txt", skiprows=[1], low_memory=False, usecols=common + elements)
    data_structure_filtered_df = data_structure_filtered_df.query("eventname == 'baseline_year_1_arm_1'")
    if all_df is None:
        all_df =  data_structure_filtered_df[["subjectkey", "interview_age", "sex"] + elements]
    else:
        all_df = all_df.merge( data_structure_filtered_df[['subjectkey'] + elements], how='outer')


# In[10]:


#check that df looks alright
all_df.head()


# In[11]:


#check if all unique entries
all_df.shape, all_df.subjectkey.unique().shape  ##some are duplicated


# In[12]:


#view duplicated entries
duplicates = (all_df[all_df.duplicated('subjectkey', keep=False)])


# In[13]:


#write duplicates to tsv to inspect
duplicates.to_csv("duplicates_df.tsv", sep="\t", index=None) #lack MRI info variables


# In[14]:


#keep duplicates that lack info for mri to remove from full dataset
duplicates = duplicates[duplicates['mri_info_manufacturer'].isna()]
duplicates.shape


# In[15]:


#delete duplicated subjects if that subject has an NA for mri_info_manufacturer
no_dup_df = (pd.merge(all_df,duplicates, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
no_dup_df.shape


# In[16]:


#write full df to .tsv file
no_dup_df.to_csv("complete_df.tsv", sep="\t", index=None)


# Next step is to clean data:
# - 

# In[17]:


# make all categorical variables 'category'
no_dup_df['site_id_l'] = no_dup_df.site_id_l.astype('category')
no_dup_df['subjectkey'] = no_dup_df.subjectkey.astype('category')
no_dup_df['rel_family_id'] = no_dup_df.rel_family_id.astype('category')
no_dup_df['sex'] = no_dup_df.sex.astype('category')
no_dup_df['ehi_y_ss_scoreb'] = no_dup_df.ehi_y_ss_scoreb.astype('category')
no_dup_df['demo_comb_income_v2'] = no_dup_df.demo_comb_income_v2.astype('category')
no_dup_df['demo_prnt_ed_v2'] = no_dup_df.demo_prnt_ed_v2.astype('category')
no_dup_df['race_ethnicity'] = no_dup_df.race_ethnicity.astype('category')
no_dup_df['demo_ed_v2'] = no_dup_df.demo_ed_v2.astype('category')
no_dup_df['medhx_ss_9b_p'] = no_dup_df.medhx_ss_9b_p.astype('category')
no_dup_df['rel_relationship'] = no_dup_df.rel_relationship.astype('category')
no_dup_df['mri_info_manufacturer'] = no_dup_df.mri_info_manufacturer.astype('category')
no_dup_df['mri_info_manufacturersmn'] = no_dup_df.mri_info_manufacturersmn.astype('category')
no_dup_df['mri_info_softwareversion'] = no_dup_df.mri_info_softwareversion.astype('category')
no_dup_df['mri_info_deviceserialnumber'] = no_dup_df.mri_info_deviceserialnumber.astype('category')
no_dup_df['mrif_score'] = no_dup_df.mrif_score.astype('category')
no_dup_df['fsqc_qc'] = no_dup_df.fsqc_qc.astype('category')
no_dup_df['imgincl_t1w_include'] = no_dup_df.imgincl_t1w_include.astype('category')
no_dup_df['imgincl_dmri_include'] = no_dup_df.imgincl_dmri_include.astype('category')


# In[18]:


#relabel to make analysis easier
no_dup_df = no_dup_df.rename(columns={'site_id_l': 'site', 'subjectkey': 'subject', 'rel_family_id': 'familyid', 'ehi_y_ss_scoreb': 'handedness', 'demo_comb_income_v2': 'household_income', 'demo_prnt_ed_v2': 'parent_ed', 'demo_ed_v2': 'subject_ed', 'medhx_ss_9b_p': 'anesthesia','rel_relationship':'siblings_twins'})


# In[19]:


#check if categorized correctly
print(no_dup_df['sex'].cat.categories)
print(no_dup_df['handedness'].cat.categories)
print(no_dup_df['household_income'].cat.categories)
print(no_dup_df['parent_ed'].cat.categories)
print(no_dup_df['race_ethnicity'].cat.categories)
print(no_dup_df['subject_ed'].cat.categories)
print(no_dup_df['mri_info_manufacturer'].cat.categories)
print(no_dup_df['mri_info_manufacturersmn'].cat.categories)
print(no_dup_df['mri_info_softwareversion'].cat.categories)
print(no_dup_df['mri_info_deviceserialnumber'].cat.categories)
print(no_dup_df['mrif_score'].cat.categories)
print(no_dup_df['fsqc_qc'].cat.categories)
print(no_dup_df['imgincl_t1w_include'].cat.categories)
print(no_dup_df['imgincl_dmri_include'].cat.categories)


# In[20]:


#relabel handedness, race_ethnicity, subject_ed
no_dup_df['handedness'] = no_dup_df['handedness'].replace([1, 2, 3], ['right','left','mixed'])
no_dup_df['handedness'] = no_dup_df.handedness.astype('category')
print(no_dup_df['handedness'].cat.categories)
#no_dup_df['handedness'] = no_dup_df.rename_categories({1: 'right', 2: 'left', 3:'mixed'}) #not sure how to do it this way

no_dup_df['race_ethnicity'] = no_dup_df['race_ethnicity'].replace([1, 2, 3, 4, 5], ['white','black','hispanic','asian','other'])
no_dup_df['race_ethnicity'] = no_dup_df.race_ethnicity.astype('category')
print(no_dup_df['race_ethnicity'].cat.categories)

no_dup_df['subject_ed'] = no_dup_df['subject_ed'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12], ['kinder','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th'])
no_dup_df['subject_ed'] = no_dup_df.subject_ed.astype('category')
print(no_dup_df['subject_ed'].cat.categories)

no_dup_df['siblings_twins'] = no_dup_df['siblings_twins'].replace([0,1,2,3], ['only_child','siblings','twins','triplets'])
no_dup_df['siblings_twins'] = no_dup_df.siblings_twins.astype('category')
print(no_dup_df['siblings_twins'].cat.categories)


# In[21]:


#replace 777, 888, and 999 with NaN
no_dup_df = no_dup_df.replace([777,888,999],np.nan)

#reindex household_income, parent_ed into correct groupings
no_dup_df['household_income'] = pd.cut(no_dup_df.household_income,bins=[0, 6, 8, 10],labels=['<50k', '>=50k & <100k', '>=100k'],include_lowest=True)
no_dup_df['parent_ed'] = pd.cut(no_dup_df.parent_ed,bins=[0, 12, 14, 17, 18, 21],labels=['< HS Diploma', 'HS Diploma/GED', 'Some College', 'Bachelor', 'Post Grad'],include_lowest=True)


# In[22]:


#check that household_income categorized correctly
print(no_dup_df['household_income'].cat.categories)
print(no_dup_df['parent_ed'].cat.categories)


# In[23]:


#check that CBCL variables are floats
no_dup_df['cbcl_scr_syn_anxdep_r'].describe() #they look good


# In[24]:


#check if column has NaN
print(no_dup_df['site'].isnull().values.any()) #none
print(no_dup_df['familyid'].isnull().values.any()) #yes


# In[25]:


#remove subjects without family ID
no_dup_df = no_dup_df[no_dup_df['familyid'].notna()]
no_dup_df.shape


# In[26]:


#keep checking columns that have NaN
print(no_dup_df['interview_age'].isnull().values.any()) #none
print(no_dup_df['sex'].isnull().values.any()) #none
print(no_dup_df['handedness'].isnull().values.any()) #none
print(no_dup_df['household_income'].isnull().values.any()) #yes


# In[27]:


#remove subjects without household_income
no_dup_df = no_dup_df[no_dup_df['household_income'].notna()]
no_dup_df.shape


# In[28]:


#keep checking columns that have NaN
no_dup_df['parent_ed'].isnull().values.any() #yes


# In[29]:


#remove subjects without parent_ed
no_dup_df = no_dup_df[no_dup_df['parent_ed'].notna()]
no_dup_df.shape


# In[30]:


#keep checking columns that have NaN
no_dup_df['race_ethnicity'].isnull().values.any() #yes


# In[31]:


#remove subjects without race_ethnicity
no_dup_df = no_dup_df[no_dup_df['race_ethnicity'].notna()]
no_dup_df.shape


# In[32]:


#keep checking columns that have NaN
print(no_dup_df['subject_ed'].isnull().values.any()) #none
print(no_dup_df['anesthesia'].isnull().values.any()) #yes


# In[33]:


#remove subjects without anesthesia
no_dup_df = no_dup_df[no_dup_df['anesthesia'].notna()]
no_dup_df.shape


# In[34]:


#keep checking columns that have NaN
print(no_dup_df['siblings_twins'].isnull().values.any()) #none


# In[35]:


#check there are NaN in CBCL/brain metrics outputs
print(no_dup_df[{'cbcl_scr_syn_anxdep_r':'cbcl_scr_syn_aggressive_r'}].isnull().values.any()) #yes


# In[36]:


#remove subjects without CBCL
no_dup_df = no_dup_df.dropna(subset=['cbcl_scr_syn_anxdep_r','cbcl_scr_syn_withdep_r','cbcl_scr_syn_somatic_r','cbcl_scr_syn_social_r','cbcl_scr_syn_thought_r','cbcl_scr_syn_attention_r','cbcl_scr_syn_rulebreak_r','cbcl_scr_syn_aggressive_r'], how='any')
no_dup_df.shape


# In[37]:


#any NAs in the brain data?
print(no_dup_df[{'mri_info_manufacturer':'smri_vol_cdk_insularh'}].isnull().values.any()) #yes

#all before mri_info complete?
print(no_dup_df[{'subject':'cbcl_scr_syn_aggressive_r'}].isnull().values.any()) #all accounted for


# In[38]:


#remove subjects without brain data, i.e. rest of dataset
no_dup_df = no_dup_df.dropna()
no_dup_df.shape


# In[39]:


#clean for QC metrics for neuroimaging data
#if dti motion is >2mm, subject should be removed
no_dup_df = no_dup_df[no_dup_df.dmri_dti_meanmotion <= 2]
#check that it looks good
no_dup_df['dmri_dti_meanmotion'].hist()
no_dup_df.shape


# In[40]:


#keep subjects with good QC (==1)
no_dup_df = no_dup_df[no_dup_df.imgincl_dmri_include == 1]
no_dup_df.shape


# In[41]:


#keep subjects with good QC (==1)
no_dup_df = no_dup_df[no_dup_df.imgincl_t1w_include == 1]
no_dup_df.shape


# In[42]:


#keep subjects with good QC (==1)
no_dup_df = no_dup_df[no_dup_df.fsqc_qc == 1]
no_dup_df.shape


# In[43]:


#keep subjects with no major incidental findings (==1 or 2)
no_dup_df = no_dup_df[(no_dup_df['fsqc_qc']==1) | (no_dup_df['fsqc_qc']==2)]
no_dup_df.shape


# In[44]:


#write cleaned dataset
no_dup_df.to_csv("cleaned_complete_df.tsv", sep="\t", index=None)


# Subset into two dataset based on the matched groups from collection 3165:
# -
# https://collection3165.readthedocs.io/en/stable/recommendations/

# In[45]:


#load in dataset that contains the IDs for the two groups
collection = pd.read_csv ('collection3165participants.tsv', sep='\t')


# In[46]:


#create list for group 1 and 2
group1 = collection[collection.matched_group == 1]
group1_subjects = group1['participant_id']
group2 = collection[collection.matched_group == 2]
group2_subjects = group2['participant_id']

#remove formatting of beginning of IDs to match the IDs from the no_dup_df
group1_subjects = group1_subjects.str[8:]
group2_subjects = group2_subjects.str[8:]

#create new column in no_dup_df to match last 11 characters in the IDs
no_dup_df['short_id'] = no_dup_df['subject'].str[5:]
print(no_dup_df['short_id'])


# In[47]:


#length of each
print(group1_subjects)
print(group2_subjects)


# In[48]:


#subset based on last 11 characters group 1
group1_df = no_dup_df[no_dup_df['short_id'].isin(group1_subjects)]
group1_df.shape


# In[49]:


#subset based on last 11 characters group 2
group2_df = no_dup_df[no_dup_df['short_id'].isin(group2_subjects)]
group2_df.shape


# In[50]:


#check that unique family ID for each subject
print(group1_df.shape, group1_df.familyid.unique().shape)  ##some are duplicated
print(group2_df.shape, group2_df.familyid.unique().shape)


# In[51]:


#randomly keep 1 participant per family
idx1 = np.random.RandomState(seed=42).permutation(np.arange(len(group1_df)))
group1_df = group1_df.iloc[idx1].drop_duplicates(subset=['familyid'])
print(group1_df.shape)

idx2 = np.random.RandomState(seed=42).permutation(np.arange(len(group2_df)))
group2_df = group2_df.iloc[idx2].drop_duplicates(subset=['familyid'])
print(group2_df.shape)


# In[52]:


#write group 1 and 2 to csv
group1_df.to_csv("group1_df.csv", index=None)
group2_df.to_csv("group2_df.csv", index=None)

