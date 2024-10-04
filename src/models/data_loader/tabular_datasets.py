import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/dataset.py 
def load_and_process_dataset(dataset, sensitive_attribute, stratify_by='sensitive', scaler=None, test_size=0.3, val_data=True, seed=42):
    if dataset == "adult":
        X, y, s = load_adult_data(sensitive_attribute=sensitive_attribute)
    elif dataset == "compas":
        X, y, s = load_compas_data(sensitive_attribute=sensitive_attribute)
    elif dataset == "census_income_kdd":
        X, y, s = load_census_income_kdd_data(sensitive_attribute=sensitive_attribute)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    
    train, val, test = process_dataset(X=X,
                                        y=y,
                                        s=s,
                                        stratify_by=stratify_by,
                                        scaler = scaler,
                                        test_size=test_size,
                                        val_data=val_data,
                                        seed=seed)
    return train, val, test

def load_dataset(dataset, sensitive_attribute='sex'):
    if dataset == "adult":
        return load_adult_data(sensitive_attribute=sensitive_attribute)
    elif dataset == "compas":
        return load_compas_data(sensitive_attribute=sensitive_attribute)
    elif dataset == "census_income_kdd":
        return load_census_income_kdd_data(sensitive_attribute=sensitive_attribute)
    else:
        raise ValueError("Dataset {} not supported".format(dataset))

def load_adult_data(path="data/tabular/adult/raw", sensitive_attribute="sex"):
    column_names = ["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native-country","target"]

    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "native-country", "education"]
    features_to_drop = ["fnlwgt"]

    df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python")
    df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python", skiprows=1)

    df = pd.concat([df_train, df_test])
    df.drop(columns=features_to_drop, inplace=True)
    
    df.dropna(inplace=True)

    if sensitive_attribute == "race":
        df = df[df["race"].isin(["White", "Black"])]
        s = df[sensitive_attribute][df["race"].isin(["White", "Black"])]
        s = (s == "White").astype(int).to_frame()
        categorical_features.append( "sex" )
    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append( "race" )

    df["target"] = df["target"].replace({"<=50K.": 0, ">50K.": 1, ">50K": 1, "<=50K": 0})
    y = df["target"].astype(int).to_frame()

    X = df.drop(columns=["target", sensitive_attribute])
    X[categorical_features] = X[categorical_features].astype("string")

    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s

# from https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/dataset.py#L321
def load_compas_data(path="data/compas/raw", sensitive_attribute="sex"):
    # We use the same features_to_keep and categorical_features from AIF360 at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/compas_dataset.py

    features_to_keep = ["sex","age","age_cat","race","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree",'c_charge_desc',"two_year_recid"]
    categorical_features = ["age_cat", "c_charge_degree", 'c_charge_desc']

    df = pd.read_csv(os.path.join(path, "compas-scores-two-years.csv"), index_col = 0)

    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[features_to_keep]

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append("race")
    elif sensitive_attribute == "race":
        s = df[sensitive_attribute].replace({
            'African-American': 0,
            'Caucasian': 1,
            'Hispanic': 2,
            'Other': 3,
            'Asian': 3,
            'Native American': 3
            })
        # s = (s == "Caucasian").astype(int).to_frame()
        s = s.astype(int).to_frame()
        categorical_features.append("sex")
    else:
        print("error")
    
    # Replace low frequency classes from c_charg_desc by a "Rest" class
    majority_classes = df['c_charge_desc'].value_counts()[df['c_charge_desc'].value_counts()>=10]
    df['c_charge_desc'] = df['c_charge_desc'].apply(lambda x: x if x in majority_classes else 'Rest')

    y = (df["two_year_recid"] ==  1 ).astype(int).to_frame()

    X = df.drop(columns=["two_year_recid", sensitive_attribute])
    
    # Convert all non-uint8 columns to float32
    X[categorical_features] = X[categorical_features].astype("string")
    string_cols = X.select_dtypes(include="string").columns
    uint8_cols = X.select_dtypes(exclude="uint8").columns
    X[list(set(uint8_cols)-set(string_cols))] = X[list(set(uint8_cols)-set(string_cols))].astype("float32")

    return X, y, s

def load_census_income_kdd_data(path="data/census_income_kdd/raw", sensitive_attribute="sex"):

    colum_names = ["age","workclass","industry_code","occupation_code","education","wage_per_hour","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","race","hispanic_origin","sex","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","capital_gains","capital_losses","dividend_from_stocks","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","weeks_worked_in_year","year","class"]


    categorical_features = [
    "workclass","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","hispanic_origin","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","year"
    ]

    feature_to_keep = [ "workclass","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","hispanic_origin","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","year"]


    df1 = pd.read_csv(os.path.join(path, "census-income.data"),header=None,names=colum_names)
    df2 = pd.read_csv(os.path.join(path, "census-income.test"),header=None,names=colum_names)



    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(keep="first", inplace=False)

    # df.columns = df.columns.str.lower()

    if sensitive_attribute == "race":
        df = df[df["race"].isin([" White", " Black"])]
        s = df[sensitive_attribute]
        s = (s == " White").astype(int).to_frame()
        categorical_features.append("sex")
    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == " Male").astype(int).to_frame()
        categorical_features.append("race")

    # # targets; 1 , otherwise 0
    y = (df["class"] == " - 50000.").astype(int).to_frame()


    # features; note that the 'target' and sentive attribute columns are dropped
    X = df.drop(columns=["class", sensitive_attribute])
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")
    return X, y, s

def process_dataset(X, y, s, stratify_by='target+sensitive', scaler='standard', test_size=0.3, val_data=True, seed=42):
    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, dtype='float32', drop_first=True)

    if stratify_by == 'target':
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size=test_size, stratify=y, random_state=seed)
    elif stratify_by == 'sensitive':
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size=test_size, stratify=s, random_state=seed)
    elif stratify_by == 'target+sensitive':
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size=test_size, stratify=y.values+s.values, random_state=seed)

    if val_data:
        if stratify_by == 'target':
            X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_train, y_train, s_train, test_size=0.1, stratify=y_train, random_state=seed)
        elif stratify_by == 'sensitive':
            X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_train, y_train, s_train, test_size=0.1, stratify=s_train, random_state=seed)
        elif stratify_by == 'target+sensitive':
            X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_train, y_train, s_train, test_size=0.1, stratify=y_train.values+s_train.values, random_state=seed)
    else:
        X_val, y_val, s_val = None, None, None

    numerical_cols = X.select_dtypes("float32").columns
    if len(numerical_cols) > 0:
        
        if scaler == 'standard':
            scaler = StandardScaler().fit(X_train[numerical_cols])

            def scale_df(df, scaler):
                return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

            X_train[numerical_cols] = X_train[numerical_cols].pipe(scale_df, scaler)
            if val_data:
                X_val[numerical_cols]   = X_val[numerical_cols].pipe(scale_df, scaler)
            X_test[numerical_cols]  = X_test[numerical_cols].pipe(scale_df, scaler)
    
    return (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test)