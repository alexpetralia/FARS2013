import pandas as pd

#####################
##   INFOGRAPHIC   ##
#####################

""" DON'T BECOME A STATISTIC: A LOOK AT FATAL ACCIDENTS IN 2013 """

# People involved in US automobile accidents
m = pd.read_pickle('fars.pkl')
m.shape[0]

# People who died in fatal automobile accidents
m[m['INJ_SEV'] == 'Fatal Injury'].shape[0]

# What weather condition was the biggest contributor of fatal accidents?
accident = pd.read_pickle('accident.pkl')
accident['WEATHER'].value_counts() # the vast majority occur during clear weather

# What lighting condition was the biggest contributor of fatal accidents?
accident['LGT_COND'].value_counts() # accidents are equally as likely during daylight and at night

# What proportion of drivers were speeding, on drugs, or drinking?
drivers = m[m['PER_TYP'] == 'Driver of a Motor Vehicle In-Transport']
tot_drivers = float(drivers.shape[0])
drivers[drivers['SPEED_DUMMY'] == 1].shape[0] / tot_drivers
drivers[drivers['DRUG_DUMMY'] == 1].shape[0] / tot_drivers
drivers[drivers['DR_DRINK'] == "Drinking"].shape[0] / tot_drivers
drivers[drivers['DR_SF'] == 1].shape[0] / tot_drivers
drivers['DR_SF1'].value_counts()

# How many drivers had at least one of these?
afd = drivers[drivers['AT_FAULT'] == 1]
not_afd = drivers[drivers['AT_FAULT'] == 0]
afd.shape[0] / tot_drivers
# 58% of drivers satisfied at least one of these conditions

# How many at-fault drivers caused all of the accidents?
len(afd['ST_CASE'].unique()) # 23,440 accidents

    # How many at-fault drivers only affected their vehicle?
len(afd[afd['VE_TOTAL'] == 1]['ST_CASE'].unique()) # 11,972 accidents

    # How many at-fault drivers hit another vehicle?
t = drivers[drivers['VE_TOTAL'] > 1].groupby('ST_CASE').sum()
t[t['AT_FAULT'] > 0].shape[0] # 11,468 accidents

# How many accidents had no at-fault drivers (ie. were unavoidable)?
x = drivers.groupby(['ST_CASE']).sum()
x[x['AT_FAULT'] == 0].shape[0] # 6,525
        # Are these 3 groups MECE? 
        # total accidents: 29,965 (see sanity check) = 11,972 + 11,468 + 6,525 (true)

# How many fatalities were due to avoidable accidents?
accident[accident['ST_CASE'].isin( afd['ST_CASE'] )]['FATALS'].sum() # 25,851
    # 79% of all those killed in fatal crashes were due to negligence

# Excluding themselves, how many people did at-fault drivers kill?
afd_died = afd[afd['INJ_SEV'] == 'Fatal Injury'].shape[0] # 15,978
tdiaa = float(accident[accident['ST_CASE'].isin( afd['ST_CASE'] )]['FATALS'].sum()) # total dead in avoidable accidents
afd_died / tdiaa # 62% at-fault drivers themselves
(tdiaa - afd_died) / tdiaa # 9,873 people, or 38% of all deaths were not-at-fault victims (drivers, passengers, etc.) killed in avoidable accidents
    # More specifically, they were:
piaa = m[m['ST_CASE'].isin( afd['ST_CASE'] )] # people in avoidable accidents
dpiaa = piaa[piaa['INJ_SEV'] == 'Fatal Injury']
dpiaa['PER_TYP'].value_counts() # not-at-fault drivers (2,401), passengers (5,459), pedestrians (1,608) and other (405) adds up to 9,873 (the number of not-at-fault victims)

# How many occupants wore safety restraints died using/not using seatbelts?
m['REST_USE'].value_counts()
m[m['REST_USE'] == 'None Used']['INJ_SEV'].value_counts()
m[m['REST_USE'] == 'Lap and Shoulder Belt Used']['INJ_SEV'].value_counts()

# In the front seat, what was your likelihood of dying given use of a safety restraint?
front = m[m['SEAT_POS'] == 'Front seat']
front['REST_USE'].value_counts() # vast majority are None Used & Lap/Shoulder belt
groups = front.groupby(['REST_USE', 'INJ_SEV']).size()
rel_groups = groups.loc[['None Used', 'Lap and Shoulder Belt Used']]
rel_groups.groupby(level = ['REST_USE']).sum()
# 27%% chance of dying if lap & shoulder belt is used
# 72% chance of dying if no restraint is used -> 45% more likely to survive in front seat

# In the backseat (second), what was your likelihood of dying given use of a safety restraint?
second = m[m['SEAT_POS'] == 'Second seat']
second['REST_USE'].value_counts() # majority are None Used & Lap/Shoulder belt
groups = second.groupby(['REST_USE', 'INJ_SEV']).size()
rel_groups = groups.loc[['None Used', 'Lap and Shoulder Belt Used']]
rel_groups.groupby(level = ['REST_USE']).sum()
# 12% chance of dying if lap & shoulder belt is used
# 40% chance of dying if no restraint is used -> 28% more likely to survive in backseat

# In the third seat, what was your likelihood of dying given use of a safety restraint?
third = m[m['SEAT_POS'] == 'Third seat']
third['REST_USE'].value_counts() # majority are None Used & Lap/Shoulder belt
groups = third.groupby(['REST_USE', 'INJ_SEV']).size()
rel_groups = groups.loc[['None Used', 'Lap and Shoulder Belt Used']]
rel_groups.groupby(level = ['REST_USE']).sum()
# 8.5% chance of dying if lap & shoulder belt is used
# 25% chance of dying if no restraint is used -> 16.5% more likely to survive in backseat