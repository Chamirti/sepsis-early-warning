VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
DEMO = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
LABEL = "SepsisLabel"
ONSET_SHIFT = 6   # SepsisLabel turns on 6 hours before actual onset