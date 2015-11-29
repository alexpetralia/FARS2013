# @author: apetralia

import os
import pandas as pd
import numpy as np

##############################
### READ IN DATABASE FILES ###
##############################

base = "C:/Users/apetralia/Desktop/FARS2013/"
csvs = base + "csv files/"
source_files = [x.split(".")[0] for x in os.listdir(csvs) if x.endswith(".csv")]

dfs = {}
for x in source_files:
    dfs[x] = pd.read_csv("%s/%s.csv" % (csvs, x))

#############################
###  CROSSWALK ON VALUES  ###
#############################

def crosswalk(dbf, file_name, delim):
    crosswalks_path = base + "/crosswalks/" + dbf + "/"
    ext = ".csv" if delim == "," else ".txt"
    key_value = np.genfromtxt(crosswalks_path + file_name + ext, delimiter = delim, dtype=None)
    pair = {k:v for k,v in key_value}
    dfs[dbf][file_name].replace(pair, inplace=True)
    
accident_replacements = {"STATE": ',', "HARM_EV": "\t", "ROAD_FNC": ',', 'MAN_COLL': ',', 'RELJCT2': ',', 'LGT_COND': ',', 'WEATHER': '\t'}
for k,v in accident_replacements.iteritems(): crosswalk('accident', k, v)
    
person_replacements = {'STATE': ',', "SEX": ",", "PER_TYP": ',', "INJ_SEV": '\t', "SEAT_POS": "\t", "REST_USE": "\t", "REST_MIS": ",", "AIR_BAG": "\t", "EJECTION": ",", "DRINKING": ',', "DRUGS": ',', "DRUGRES1": '\t', "DRUGRES2": '\t', "DRUGRES3": '\t', "DOA": ',', 'WORK_INJ': ',', 'RACE': '\t', 'HISPANIC': '\t', 'LOCATION': '\t'}
for k,v in person_replacements.iteritems(): crosswalk('person', k, v)

vehicle_replacements = {'STATE': ',', 'HIT_RUN': ',', 'OWNER': ',', 'MAKE': '\t', 'BODY_TYP': '\t', 'SPEC_USE': ',', 'ROLINLOC': ',', 'DEFORMED': ',', 'M_HARM': '\t', 'FIRE_EXP': ',', 'DR_DRINK': ',', 'L_STATUS': ',', 'SPEEDREL': '\t', 'VALIGN': ',', 'VSURCOND': '\t', 'DR_SF1': '\t', 'DR_SF2': '\t', 'DR_SF3': '\t', 'DR_SF4': '\t'}
for k,v in vehicle_replacements.iteritems(): crosswalk('vehicle', k, v)
    
""" ADD BODY_SIZE FEATURE VIA CROSSWALK """
dfs['vehicle']['BODY_SIZE'] = dfs['vehicle']['BODY_TYP']
crosswalks_path = base + "/crosswalks/vehicle/BODY_SIZE.txt"
key_value = np.genfromtxt(crosswalks_path, delimiter = '\t', dtype=None)
dfs['vehicle']['BODY_SIZE'].replace({str(k):v for k,v in key_value}, inplace=True)

####################
###  FORMATTING  ###
####################
    
""" CONVERT TO DATETIME64 """
def join_datetimes(dbf, colname = 'crash_', timeinfo = ['DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE']):
    day, month, year, hour, minute = timeinfo
    colname = colname + 'datetime'
    
    # convert date and time columns into datetimes
    datetimes = dfs[dbf][timeinfo].applymap(str)
    datetimes.replace({'99': np.nan, '9999': np.nan, '88': np.nan, '8888': np.nan}, inplace=True)
    datetimes[colname] = ( 
        datetimes[day] + "/" + datetimes[month] + "/" + datetimes[year] + " " + 
        datetimes[hour] + ":" + datetimes[minute] )
    datetimes[colname] = pd.to_datetime(datetimes[colname], format="%d/%m/%Y %H:%M")
    
    # for those without timestamps, determine date only (will convert to 00:00 time)
    dates = datetimes[pd.isnull(datetimes[colname])]
    dates[colname] = datetimes[day] + "/" + datetimes[month] + "/" + datetimes[year]
    dates[colname] = pd.to_datetime(dates[colname], format="%d/%m/%Y")
    
    # combine the two datetime dataframes and join back to dbf
    datetimes = datetimes[pd.notnull(datetimes[colname])]
    datetimes = datetimes.append(dates)
    dfs[dbf] = dfs[dbf].join(datetimes[colname])
    
join_datetimes('accident')
join_datetimes('person', 'fatality_', ['DEATH_DA', 'DEATH_MO', 'DEATH_YR', 'DEATH_HR', 'DEATH_MN'])

""" REPLACE NULL VALUES IN EXPLICITLY IDENTIFIED COLUMNS AS NP.NAN """
def nullify(dbf, columns = [], null_equivs = []):
    for col, nulls in zip(columns, null_equivs):
        null_vals = {k:np.nan for k in nulls}
        dfs[dbf][col].replace(null_vals, inplace=True)

nullify('vehicle', ['TRAV_SP', 'VSPD_LIM', 'DR_HGT', 'DR_WGT', 'NUMOCCS', 'MOD_YEAR', 'PREV_ACC', 'PREV_SUS', 'PREV_DWI', 'PREV_SPD'], [[997, 998, 999], [98, 99], [998, 999], [997, 998, 999], [96, 98, 99], [9998, 9999], [98, 99, 998], [99, 998], [99, 998], [99, 998]])
nullify('person', ['AGE', 'ALC_RES'], [[998, 999], range(95,100)])
nullify('accident', ['LATITUDE', 'LONGITUD'], [[77.7777, 88.8888, 99.9999], [777.7777, 888.8888, 999.9999]])
# remember: convert all 999/99/98/88/888s to NaNs before computing summary stats (check each field before summary stats)

""" CLEAN VEHICLE['SEAT_POS'] """
for x in ['Front', 'Second', 'Third', 'Fourth']:
    dfs['person']['SEAT_POS'][dfs['person']['SEAT_POS'].str.contains(x)] = x + ' seat'
dfs['person']['SEAT_POS'][~dfs['person']['SEAT_POS'].str.contains('Front|Second|Third|Fourth') & ~(dfs['person']['SEAT_POS'] == 'Not a Motor Vehicle Occupant')] = 'Other'

""" CREATE UNIQUE CASE-VEHICLE IDENTIFIERS FOR SIMPLER JOINS """
for x in ['vehicle', 'person']:
    dfs[x]['UID'] = dfs[x]['ST_CASE'].astype(str) + "-" + dfs[x]['VEH_NO'].astype(str)
    
""" COMBINE LATITUDE AND LONGITUDES """
dfs['accident']['COORDINATES'] = dfs['accident']['LATITUDE'].astype(str) + ', ' + dfs['accident']['LONGITUD'].astype(str)
    
##################
###  DROPPING  ###
##################

dfs['accident'].drop(['CF1', 'CF2', 'CF3', 'DAY_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE', 'TWAY_ID', 'TWAY_ID2', 'MILEPT', 'SP_JUR', 'PVH_INVL', 'VE_FORMS', 'REL_ROAD', 'SCH_BUS', 'RAIL', 'WRK_ZONE', 'NHS', 'ROUTE', 'TYP_INT', 'PERNOTMVIT', 'PERMVIT', 'COUNTY', 'CITY', 'DAY_WEEK', 'WEATHER1', 'WEATHER2', 'NOT_HOUR', 'NOT_MIN', 'ARR_HOUR', 'ARR_MIN', 'HOSP_HR', 'HOSP_MN', 'RELJCT1', 'LATITUDE', 'LONGITUD'], axis=1, inplace=True)

dfs['person'].drop(['VE_FORMS', 'STR_VEH', 'COUNTY', 'DAY', 'MONTH', 'HOUR', 'MINUTE', 'ROAD_FNC', 'HARM_EV', 'MAN_COLL', 'SCH_BUS', 'MAKE', 'MAK_MOD', 'BODY_TYP', 'MOD_YEAR', 'TOW_VEH', 'SPEC_USE', 'EMER_USE', 'ROLLOVER', 'IMPACT1', 'FIRE_EXP', 'EJ_PATH', 'EXTRICAT', 'ALC_STATUS', 'ATST_TYP', 'HOSPITAL', 'DEATH_TM', 'DEATH_DA', 'DEATH_MO', 'DEATH_YR', 'DEATH_HR', 'DEATH_MN', 'P_SF1', 'P_SF2', 'P_SF3', 'LAG_HRS', 'LAG_MINS', 'CERT_NO', 'DRUGTST1', 'DRUGTST2', 'DRUGTST3', 'DSTATUS', 'DRUG_DET', 'ALC_DET'], axis=1, inplace=True)

dfs['vehicle'].drop(['VE_FORMS', 'HARM_EV', 'MAN_COLL', 'DAY', 'MONTH', 'HOUR', 'MINUTE', 'UNITTYPE', 'REG_STAT', 'MODEL', 'VIN', 'TOW_VEH', 'J_KNIFE', 'MCARR_ID', 'MCARR_I1', 'MCARR_I2', 'GVWR', 'V_CONFIG', 'CARGO_BT', 'HAZ_INV', 'HAZ_PLAC', 'HAZ_ID', 'HAZ_CNO', 'HAZ_REL', 'BUS_USE', 'EMER_USE', 'UNDERIDE', 'ROLLOVER', 'IMPACT1', 'TOWED', 'VEH_SC1', 'VEH_SC2', 'MAK_MOD', 'VIN_1', 'VIN_2', 'VIN_3', 'VIN_4', 'VIN_5', 'VIN_6', 'VIN_7', 'VIN_8', 'VIN_9', 'VIN_10', 'VIN_11', 'VIN_12', 'DR_PRES', 'L_STATE', 'DR_ZIP', 'L_TYPE', 'CDL_STAT', 'L_ENDORS', 'L_COMPL', 'L_RESTRI', 'PREV_OTH', 'FIRST_MO', 'FIRST_YR', 'LAST_MO', 'LAST_YR', 'VTRAFWAY', 'VNUM_LAN', 'VPROFILE', 'VPAVETYP', 'VTRAFCON', 'VTCONT_F', 'P_CRASH1', 'P_CRASH2', 'P_CRASH3', 'PCRASH4', 'PCRASH5', 'ACC_TYPE'], axis=1, inplace=True)

###############################
###   CONVENIENCE METHODS   ###
###############################

accident = dfs['accident'].copy()
person = dfs['person'].copy()
vehicle = dfs['vehicle'].copy()

def query(df, case_id):
    x = df[df['ST_CASE'] == case_id]
    print x
    return x

##########################
###   PRE-PROCESSING   ###
##########################

# Create a merged dataset between person, vehicle and accident
persons = person.drop(['LOCATION', 'WORK_INJ', 'DOA', 'EJECTION', 'AIR_BAG', 'REST_MIS', 'DRINKING'], axis=1)
vehicle_rel_cols = ['UID', 'NUMOCCS', 'MAKE', 'BODY_TYP', 'BODY_SIZE', 'MOD_YEAR', 'TRAV_SP', 'DEFORMED', 'M_HARM', 'DR_HGT', 'DR_WGT', 'PREV_ACC', 'PREV_SUS', 'PREV_DWI', 'PREV_SPD', 'SPEEDREL', 'DR_SF1', 'DR_SF2', 'DR_SF3', 'DR_SF4', 'VSPD_LIM', 'VSURCOND', 'DEATHS', 'DR_DRINK']
merged = persons.merge(vehicle[vehicle_rel_cols], how='left', on='UID')
accident_rel_cols = ['ST_CASE', 'VE_TOTAL', 'PERSONS', 'COORDINATES', 'WEATHER', 'LGT_COND', 'RELJCT2', 'crash_datetime']
m = merged.merge(accident[accident_rel_cols], how='left', on='ST_CASE')

#*# Define an at-fault driver (see explanation below in sanity checks)
""" Criteria: driver AND
    (1) drinking (either DR_DRINK == 'Drinking' or ALC_RES > 8), 
    (2) on drugs (either DRUGS.str.contains("Yes") or DRUGRES includes drugs), 
    (3) speeding (either TRAV_SP>VSPD_LIM+10 or SPEEDREL.str.contains('Yes')), or 
    (4) DR_SF attributed 
If multiple drivers fit these conditions, then both are jointly at fault for the accident """

    # Overhead
not_drug_violations = [x for x in m.DRUGRES1.unique() if ('No' in x or 'Unknown' in x and 'Positive' not in x)]
drug_violations = [x for x in m.DRUGRES1.unique() if x not in not_drug_violations]
map(lambda x: m[x].replace({'Driver has a Driving Record or Driver\'s License from More than One State': "None", 'Unknown': "None"}, inplace=True), ['DR_SF1', 'DR_SF2', 'DR_SF3', 'DR_SF4']) # recode insignificant violation as None
m['DR_SF'] = 0

    # Redefine inconsistent data
def correct_drugs(df):
    if (any([y in df['DRUGRES1'] for y in drug_violations]) or any([y in df['DRUGRES2'] for y in drug_violations]) or any([y in df['DRUGRES3'] for y in drug_violations])) and ('no' in df['DRUGS'].lower()):
        df['DRUGS'] = 'Yes (recoded)'
    return df
def correct_speed(df):
    if (df['TRAV_SP'] > (df['VSPD_LIM']+10)) and ('Yes' not in df['SPEEDREL']):
        df['SPEEDREL'] = 'Yes (recoded)'
    if pd.isnull(df['SPEEDREL']):
        df['SPEEDREL'] = 'Unknown (recoded)'
    return df
def consolidate_DRSF(df):
    condition = any(map(lambda x: x != "None", [df['DR_SF1'], df['DR_SF2'], df['DR_SF3'], df['DR_SF4']]))
    return 1 if condition else 0
# DR_DRINK is directly based on ALC_RES -> coded correctly
m = m.apply(correct_drugs, axis=1)
m = m.apply(correct_speed, axis=1)

    # Add dummies
m['DRUG_DUMMY'], m['SPEED_DUMMY'] = 0, 0
m['DRUG_DUMMY'][m['DRUGS'].str.contains('Yes')] = 1
m['SPEED_DUMMY'][m['SPEEDREL'].str.contains('Yes')] = 1
m['DR_SF'] = m.apply(consolidate_DRSF, axis=1)

    # Function to determine fault
def determine_fault(df):
    condition_1 = df['DR_DRINK'] == "Drinking" # checked: always coincides with illegal BAC
    condition_2 = df['DRUG_DUMMY'] == 1
    condition_3 = df['SPEED_DUMMY'] == 1
    condition_4 = df['DR_SF'] == 1
    return 1 if (condition_1 or condition_2 or condition_3 or condition_4) else 0
    
m['AT_FAULT'] = m.apply(determine_fault, axis=1)

# Serialize the dataframes for quick access
m.to_pickle('fars.pkl')
accident.to_pickle('accident.pkl') # used in Infographic

#########################
###   SANITY CHECKS   ###
#########################

""" .pdf = "Alcohol-Impaired Driving". Traffic Safety Facts: 2013 Data. NHTSA (Dec. 2014). """

# How many fatalities were there as a result of drunk drivers?
""" .pdf: 10,076 """
accident[accident['DRUNK_DR'] > 0]['FATALS'].sum() # 9,946
accident[accident['ST_CASE'].isin(vehicle[vehicle['DR_DRINK'] == 'Drinking']['ST_CASE'])]['FATALS'].sum() # 9,946

# How many drunk drivers are in the dataset? Check person['DRINKING'] for PER_TYP: drivers
person.query('PER_TYP == "Driver of a Motor Vehicle In-Transport" and DRINKING == "Yes (Alcohol Involved)"').shape[0] # 6,685. What if they are classified as Unknown or Not Reported?
person.query('PER_TYP == "Driver of a Motor Vehicle In-Transport" and DRINKING != "No (Alcohol Not Involved)"').shape[0] # 21,138

# Is person['DRINKING'] reliable? Check against another dataframe
vehicle[vehicle['DR_DRINK'] == 'Drinking'].shape[0] # 9,250 drunk drivers here

# Can we find all people who are PER_TYP: driver & in the same vehicle as vehicle['DR_DRINK']?
person[person['UID'].isin(vehicle[vehicle['DR_DRINK'] == 'Drinking']['UID']) & (person['PER_TYP'] == 'Driver of a Motor Vehicle In-Transport')].shape[0] # 9,250: an exact match. In other words, vehicle['DR_DRINK'] is based on the previous subset criteria from the person dataframe. 

# How are we sure that person['DRINKING'] is unreliable? Check an example
rel_cols = ['DRINKING', 'ALC_RES', 'UID']
query(vehicle, 10035).merge(person[rel_cols], how='left', on='UID') # here, DR_DRINK is coded as "Drinking", DRINKING is coded "No" and ALC_RES is .23. A positive BAC result means that DRINKING was coded incorrectly.

# Using vehicle['DR_DRINK'], how many drunk drivers died in their own accidents?
""" .pdf: 6,515 """
person[person['UID'].isin(vehicle[vehicle['DR_DRINK'] == 'Drinking']['UID']) & (person['PER_TYP'] == 'Driver of a Motor Vehicle In-Transport') & (person['INJ_SEV'] == 'Fatal Injury')].shape[0] # 6,731

# Could a driver have drunk but it is not recorded as a DR_SF?
t = vehicle[vehicle['DR_DRINK'] == 'Drinking'] # yes

# Could a driver have been speeding but it is not recorded as a DR_SF?
t = vehicle[vehicle['SPEEDREL'] != 'No'] # yes

# Could a driver have been on drugs but it is not recorded as a DR_SF?
drugged = drivers[drivers['DRUGS'] == 'Yes (Drugs Involved)']
rel_cols = ['ST_CASE', 'VEH_NO', 'DR_SF1', 'DR_SF2', 'DR_SF3', 'DR_SF4', 'SPEEDREL', 'DR_HGT', 'DR_WGT', 'NUMOCCS']
drugged_drivers = pd.merge(drugged, vehicle[rel_cols], how='left', on=['ST_CASE', 'VEH_NO'])
t = drugged_drivers[drugged_drivers['DR_SF1'] == "None"] # yes
    # THEREFORE: must expand definition of 'at-fault' beyond DR_SF
    # ASSUMPTION: DR_SF, speeding, on drugs or intoxicated imply fault in an accident. this may not always be true; for example, one could be intoxicated and crash while the real cause of the crash is mechanical failure. this driver would be incorrectly categorized as at fault when in reality, it was the mechanical failure at fault

# Can a driver have sped but not have SPEEDREL as exceeded speed limit?
vehicle[(vehicle['TRAV_SP'] - vehicle['VSPD_LIM']) > 0].head()
query(vehicle, 10011) # yes

# Can a driver have SPEEDREL as exceeded speed limit but not have TRAV_SP recorded?
vehicle[vehicle['SPEEDREL'].str.contains('Yes')]['TRAV_SP'].value_counts(dropna=False) # yes

# Can DRUG_RES be coded when DRUGS is not?
query(person, 10008) # yes

# Can ALC_RES be coded when DRINKING is listed as 'No' or other?
person[(person['ALC_RES'] > 40)].head(40) # yes
query(person, 350247)

# Can DRINKING be "Yes" when BAC is within legal limits?
dx[(dx['ALC_RES'] < 5) & (dx['ALC_RES'] > 0)].head(30)
query(person, 50406) # yes. this may be an error in data - assumed correct because DRINKING flows into the aggregate accident['DRUNK_DR'] statistics

# Can DR_DRINK be "No" when BAC is above 8?
m[(m['DR_DRINK'] == 'No Drinking') & (m['ALC_RES'] > 8)].shape[0] # no

# Does each accident require a driver? In other words, should all the accidents that contain at least one driver add up to the total number of accidents?
accident.shape[0] # total accidents (30,057)
len(drivers['ST_CASE'].unique()) # number of accidents with drivers (29,965); they are NOT equal
    # Are there accidents that contain no drivers?
l = accident[~accident['ST_CASE'].isin( drivers['ST_CASE'] )] # yes, 92 accidents
m[m['ST_CASE'].isin(l['ST_CASE'])].head(20) # check actual cases: true
# therefore, accidents with drivers should add up to 29,965, not 30,057


""" The accuracy of the data must be questioned. Some internal checks do not match and there is a lot of missing data (eg. "Yes exceeded speed limit" when no speed limit is recorded and no SPEEDREL when speed clearly exceeds speed limit). """
