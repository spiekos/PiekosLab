#' *Piekos Lab Rotation*
#' *Ashley Trocle*
#' *Spring 2026 Rotation*

# Load libraries
library(readxl)
library(dplyr)
library(janitor)
library(table1)
library(Hmisc)

filepath <- "C:\\Users\\atrocle\\Documents\\PhD\\Rotations\\Piekos\\raw_data\\dp3 master table v2.xlsx"  
sheet1 <- read_excel(filepath, sheet = "Sheet1") |>
  clean_names() |>
  mutate(
    group = ifelse(id == "DP3-0343" & group == "sptb", "sPTB", group),
    infant_sex = case_when(
        infant_sex == "f" ~ "F",  
        infant_sex == "m" ~ "M",
        TRUE ~ infant_sex
      )
    )

#Sheet 1 data 

meta_ids <- sheet1 |>
  filter(status %in% c("Delivered", "Delivered off site")) |>
  transmute(
    patient_id = id,
    group = factor(group, levels = c("Control","FGR","HDP","sPTB")),
    omics_subcohort = omics_study_n_133,
    birthweight = birthweight,
    gestational_age = gest_age_del,
    fetal_sex = factor(infant_sex,
                       levels = c("F","M"), 
                       labels = c("Female","Male")),
    risk_factor_yesno = factor(risk_factor_y_n, 
                               levels = c(0,1),
                               labels = c("No","Yes")),
    risk_factor_type = risk_factor_type,
    subgroup = subgroup
  )

#confirm cohort

nrow(meta_ids)
table(meta_ids$group)
table(meta_ids$omics_subcohort, useNA = "ifany")

voi <- read_excel(filepath, sheet = "variables of interest") |>
  clean_names()

#variables of interest sheet

meta_voi <- voi |>
  filter(status %in% c("Delivered", "Delivered off site")) |>
  transmute(
    patient_id = id,
    #group = factor(group, levels = c("Control","FGR","HDP","sPTB")),
    maternal_age = maternal_age,
    height_cm = height_cm,                 
    weight_kg = weight_kg,                 
    delivery_bmi= delivery_bmi,
    prepreg_weight= prepregnancy_weight_self_or_record,
    prepreg_bmi = prepregnancy_bmi_self_or_record,
    maternal_race = race,
    maternal_ethnicity = ethnicity,
    smoking = smoking,
    gravida = gravida,
    parity = parity,
    diabetes = diabetes,
    chronic_htn = chtn,
    labor_onset = labor_onset,
    delivery_route = route_of_delivery_1_vag_2_cs,
    apgar_1 = apgar_1,
    apgar_5 = apgar_5,
    nicu_days = nicu_days,
    aspirin_use = aspirin
  ) |>
  mutate(
    # maternal_race: 1 = White, 2 = Black, 3 = Asian, 99 = Missing
    maternal_race = case_when(
      maternal_race %in% c("AFRICIAN AMERICAN", "BLACK") ~ 2,
      maternal_race %in% c("ASIAN", "KOREAN") ~ 3,
      maternal_race %in% c("WHITE") ~ 1,
      maternal_race %in% c("DECLINED", "UNKNOWN", "na", NA) ~ 99,
      TRUE ~ NA_real_
    ),
    maternal_race = factor(
      maternal_race,
      levels = c(1, 2, 3, 99),
      labels = c("White", "Black", "Asian", "Missing")
    ),
    # maternal_ethnicity: 0 = Non-Hispanic, 1 = Hispanic, 99 = Missing
    maternal_ethnicity = case_when(
      maternal_ethnicity %in% c("HISPANIC OR LATINO") ~ 1,
      maternal_ethnicity %in% c("NOT HISPANIC OR LATINO", "Non Hispanic") ~ 0,
      maternal_ethnicity %in% c("DECLINED", "Unknown", "UNSPECIFIED", "na", NA) ~ 99,
      TRUE ~ NA_real_
    ),
    maternal_ethnicity = factor(
      maternal_ethnicity,
      levels = c(0, 1, 99),
      labels = c("Non-Hispanic", "Hispanic", "Missing")
    ),
    # smoking: 0 = Never, 1 = Quit, 2 = Yes, 99 = Missing
    smoking = case_when(
      smoking == "Never" ~ 0,
      smoking == "Quit"  ~ 1,
      smoking == "Yes"   ~ 2,
      smoking %in% c("na", NA) ~ 99,
      TRUE ~ NA_real_
    ),
    smoking = factor(
      smoking,
      levels = c(0, 1, 2, 99),
      labels = c("Never", "Quit", "Yes", "Missing")
    ),
    # diabetes: 0 = No diabetes, 1 = GDM, 2 = Type I, 3 = Type II, 99 = Missing
    diabetes = case_when(
      diabetes == 0 ~ 0,
      diabetes == 1 ~ 1,
      diabetes == 2 ~ 2,
      diabetes == 3 ~ 3,
      diabetes %in% c(NA, "na") ~ 99,
      TRUE ~ NA_real_
    ),
    diabetes = factor(
      diabetes,
      levels = c(0, 1, 2, 3, 99),
      labels = c("No diabetes","Gestational diabetes", "Type I", "Type II", "Missing")
    ),
    # chronic_htn: 0 = No, 1 = Yes, 99 = Missing
    chronic_htn = case_when(
      chronic_htn == 0 ~ 0,
      chronic_htn == 1 ~ 1,
      chronic_htn %in% c(NA, "na") ~ 99,
      TRUE ~ NA_real_
    ),
    chronic_htn = factor(
      chronic_htn,
      levels = c(0, 1, 99),
      labels = c("No", "Yes", "Missing")
    ),
    # labor onset: 0 = spontaneous, 1= induced, 2 = PROM, 99 = Missing
    labor_onset = case_when(
      labor_onset %in% c("0", "Spontaneous") ~ 0, 
      labor_onset %in% c("1", "Induced", "Induced (does not include PROM") ~ 1,
      labor_onset %in% c("2", "PROM") ~ 2,
      labor_onset %in% c("?", "na", NA) ~ 99,
      TRUE ~ 99 
    ),
    labor_onset = factor(
      labor_onset,
      levels = c(0, 1, 2, 99),
      labels = c("Spontaneous", "Induced", "PROM", "Missing")
    ),
    # delivery_route: 1 = Vaginal, 2 = Cesarean, 99 = Missing
    delivery_route = case_when(
      delivery_route %in% c("1", "2") ~ delivery_route,
      delivery_route %in% c("0", "chart", "na", NA) ~ "99",
      TRUE ~ NA_character_
    ),
    delivery_route = as.numeric(delivery_route),
    delivery_route = factor(
      delivery_route,
      levels = c(1, 2, 99),
      labels = c("Vaginal", "Cesarean", "Missing")
    )
  )

nrow(meta_voi)


clinical <- read_excel(filepath, sheet = "clinical data", skip  = 1) |>
  clean_names()

meta_clinical <- clinical |>
  filter(status %in% c("Delivered", "Delivered off site")) |>
  transmute(
    patient_id = id,
    adi_nat_rank = adi_national_rank,
    newborn_race = newborn_race,
    nicu_indication = nicu_indication_yn,
    newborn_los = newborn_total_length_of_stay,
    insurance_group = hosp_insurance_grping
  ) |>
  mutate(
    # newborn_race:1 = White, 2 = Black, 3 = Asian, 4 = American Indian,
    #              5 = Other, 9 = Missing
    newborn_race = case_when(
      newborn_race %in% c("WHITE") ~ 1,
      newborn_race %in% c("BLACK") ~ 2,
      newborn_race %in% c("ASIAN", "OTH ASIAN") ~ 3,
      newborn_race %in% c("AMERICAN INDIAN") ~ 4,
      newborn_race %in% c("OTHER") ~ 5,
      newborn_race %in% c("NOT SPECIFIED", "Missing", NA) ~ 99,
      TRUE ~ 99
    ),
    newborn_race = factor(
      newborn_race,
      levels = c(1, 2, 3, 4, 5, 99),
      labels = c("White", "Black", "Asian", "American Indian", "Other",
                 "Missing")
      ),
    nicu_indication = case_when(
      nicu_indication %in% c("N") ~ 0,
      nicu_indication %in% c("Y") ~ 1,
      nicu_indication %in% c("Missing", "na", NA) ~ 99,
      TRUE ~ NA_real_
    ),
    nicu_indication = factor(
      nicu_indication,
      levels = c(0, 1, 99),
      labels = c("No", "Yes", "Missing")
    )
  )
nrow(meta_clinical)

#Join
meta_all <- meta_ids |>
  left_join(meta_voi,      by = c("patient_id")) |>
  left_join(meta_clinical, by = "patient_id")

meta_all <- meta_all |>
  mutate(
    birthweight = na_if(birthweight, "delivered off site"),
    birthweight = as.numeric(birthweight),
    birthweight_cat = cut(
      birthweight,
      breaks = c(-Inf, 2500, 3000, 3500, Inf),
      labels = c("<2500", "2500–2999", "3000–3499", "≥3500"),
      right = FALSE
    ),
    maternal_age = as.numeric(maternal_age),
    maternal_age = if_else(maternal_age == 0, NA_real_, maternal_age),
    maternal_age_cat = cut(
      maternal_age,
      breaks = c(18, 25, 30, 35, 40, 50),  
      labels = c("18–24", "25–29", "30–34", "35–39", "40+")
    ),
    gestational_age = as.numeric(gestational_age),
    gestational_age_cat = cut(gestational_age,
      breaks = c(-Inf, 34, 37, 40, Inf),
      labels = c("<34", "34–36.9", "37–39.9", "≥40")
    ),
    prepreg_bmi = na_if(prepreg_bmi, "na"),
    prepreg_bmi = as.numeric(prepreg_bmi),
    prepreg_bmi_cat = cut(prepreg_bmi,
      breaks = c(-Inf, 18.5, 25, 30, 40, Inf),
      labels = c("Underweight", "Normal", "Overweight", 
                 "Obese", "Severely Obese")               
    ),
    gravida = as.numeric(gravida),
    gravida_cat = case_when(
      gravida == 0 ~ 0,                
      gravida >= 1 & gravida <= 5 ~ 1, 
      gravida >= 6 ~ 2,             
      TRUE ~ NA_real_
    ),
    gravida_cat = factor(
      gravida_cat,
      levels = c(0, 1, 2),
      labels = c("Nulligravidity (0)", "Low multigravidity (1–5)", 
                 "Grand multigravidity (≥6)")
    ),
    parity = na_if(parity, "na"),
    parity = na_if(parity, "NA"),
    parity = na_if(parity, "Missing"),
    parity = as.numeric(parity),
    parity_cat = case_when(
      parity == 0 ~ 0,                  
      parity >= 1 & parity <= 3 ~ 1,    
      parity >= 4 ~ 2,                  
        TRUE ~ NA_real_
      ),
    parity_cat = factor(
      parity_cat,
      levels = c(0, 1, 2),
      labels = c("Nulliparity (0)", "Low multiparity (1–3)",
                 "Grand multipara (≥4)"
                 )
    ),
    apgar_1 = na_if(apgar_1, "na"),
    apgar_1 = as.numeric(apgar_1),  
    apgar_1_cat = cut(
      apgar_1,
      breaks = c(-Inf, 3, 6, 10),
      labels = c("0–3", "4–6", "7–10"),
      right = TRUE,              # 0–3, 4–6, 7–10
      include.lowest = TRUE
    ),
    apgar_5 = na_if(apgar_5, "na"),
    apgar_5 = as.numeric(apgar_5),   
    apgar_5_cat = cut(
      apgar_5,
      breaks = c(-Inf, 3, 6, 10),
      labels = c("0–3", "4–6", "7–10"),
      right = TRUE,             
      include.lowest = TRUE
    ),
    newborn_los = as.numeric(newborn_los),
    newborn_los_cat = case_when(
      newborn_los < 7  ~ 0,
      newborn_los >= 7 ~ 1,
      TRUE ~ NA_real_
    ),
    newborn_los_cat = factor(
      newborn_los_cat,
      levels = c(0, 1),
      labels = c("<7 days", "≥7 days")
    )  
  )

#Variable Labels
labels <- c(
  birthweight = "Birthweight (g) - Continuous",  
  birthweight_cat = "Birthweight (g) - Catagorical",              
  gestational_age_cat = "Gestational age (weeks)",          
  fetal_sex = "Fetal Sex",                
  risk_factor_yesno = "Presence of Risk Factor",         
  maternal_age_cat = "Maternal age (years)",               
  prepreg_bmi_cat = "Pregravid BMI",                
  maternal_race = "Maternal Race",          
  maternal_ethnicity = "Maternal Ethnicity",          
  smoking = "Smoking",                  
  gravida_cat = "Gravida",                   
  parity_cat = "Parity",                   
  diabetes = "Diabetes",                  
  chronic_htn = "Chronic Hypertension",                 
  labor_onset = "Labor Onset",                
  delivery_route = "Route of Delivery",             
  apgar_1_cat = "APGAR 1", 
  apgar_5_cat = "APGAR 5",
  aspirin_use = "Aspirin Use", 
  adi_nat_rank = "ADI National Rank",
  newborn_race = "Newborn Race", 
  nicu_indication = "NICU Indication",            
  newborn_los_cat = "Newborn Length of Stay (days)- Catagorical",
  newborn_los = "Newborn Length of Stay (days) - Continuous",
  insurance_group = "Hospital Insurance Grouping"
)

for (v in names(labels)) {
  Hmisc::label(meta_all[[v]]) <- labels[[v]]
}

#Pooled
meta_all <- meta_all |>
  mutate(
    group_pooled = if_else(group == "Control", "Control", "Pooled"),
    group_pooled = factor(group_pooled, levels = c("Control", "Pooled"))
  )

#Omics

meta_omics <- meta_all |>
  filter(omics_subcohort == 1)  

nrow(meta_omics)
table(meta_omics$group, useNA = "ifany") 

#Table One
pvalue <- function(x, ...) {
  y <- unlist(x)
  g <- factor(rep(seq_along(x), times = sapply(x, length)))
  if (is.numeric(y)) {
    if (nlevels(g) == 2) {
      p <- t.test(y ~ g)$p.value
    } else {
      p <- anova(lm(y ~ g))$`Pr(>F)`[1]
    }
  } else {
    p <- suppressWarnings(chisq.test(table(y, g))$p.value)
  }
  c("", format.pval(p, digits = 3, eps = 0.001))
}

table_one <- table1(
  ~ birthweight_cat +
    birthweight + 
    gestational_age_cat +            
    fetal_sex +                 
    risk_factor_yesno +           
    maternal_age_cat +                
    prepreg_bmi_cat +                 
    maternal_race +             
    maternal_ethnicity +          
    smoking +                     
    gravida_cat +                     
    parity_cat +                      
    diabetes +                   
    chronic_htn +                 
    labor_onset +                 
    delivery_route +              
    apgar_1_cat +                     
    apgar_5_cat +                     
    aspirin_use +                  
    adi_nat_rank +                 
    newborn_race +                
    nicu_indication +             
    newborn_los_cat + 
    newborn_los +
    insurance_group                
  | group,
  data = meta_all,
  render.continuous = c("Median [Q1, Q3]" = "MEDIAN [Q1, Q3]"),
  extra.col = list(`P value` = pvalue),
  overall = FALSE
)

table_one

#Table 1 Omics
table_one_omics <- table1(
  ~ birthweight_cat +
    birthweight + 
    gestational_age_cat +            
    fetal_sex +                 
    risk_factor_yesno +           
    maternal_age_cat +                
    prepreg_bmi_cat +                 
    maternal_race +             
    maternal_ethnicity +          
    smoking +                     
    gravida_cat +                     
    parity_cat +                      
    diabetes +                   
    chronic_htn +                 
    labor_onset +                 
    delivery_route +              
    apgar_1_cat +                     
    apgar_5_cat +                     
    aspirin_use +                  
    adi_nat_rank +                 
    newborn_race +                
    nicu_indication +             
    newborn_los_cat + 
    newborn_los +
    insurance_group                
  | group,
  data = meta_omics,
  render.continuous = c("Median [Q1, Q3]" = "MEDIAN [Q1, Q3]"),
  extra.col = list(`P value` = pvalue),
  overall = FALSE
)

table_one_omics

#Table 1 Pooled

table_one_pooled <- table1(
  ~ birthweight_cat +
    birthweight + 
    gestational_age_cat +            
    fetal_sex +                 
    risk_factor_yesno +           
    maternal_age_cat +                
    prepreg_bmi_cat +                 
    maternal_race +             
    maternal_ethnicity +          
    smoking +                     
    gravida_cat +                     
    parity_cat +                      
    diabetes +                   
    chronic_htn +                 
    labor_onset +                 
    delivery_route +              
    apgar_1_cat +                     
    apgar_5_cat +                     
    aspirin_use +                  
    adi_nat_rank +                 
    newborn_race +                
    nicu_indication +             
    newborn_los_cat + 
    newborn_los +
    insurance_group                
  | group_pooled,
  data = meta_all,
  render.continuous = c("Median [Q1, Q3]" = "MEDIAN [Q1, Q3]"),
  extra.col = list(`P value` = pvalue),
  overall = FALSE
)

table_one_pooled

#Table 1 Pooled Omics

table_one_omics_pooled <- table1(
  ~ birthweight_cat +
    birthweight + 
    gestational_age_cat +            
    fetal_sex +                 
    risk_factor_yesno +           
    maternal_age_cat +                
    prepreg_bmi_cat +                 
    maternal_race +             
    maternal_ethnicity +          
    smoking +                     
    gravida_cat +                     
    parity_cat +                      
    diabetes +                   
    chronic_htn +                 
    labor_onset +                 
    delivery_route +              
    apgar_1_cat +                     
    apgar_5_cat +                     
    aspirin_use +                  
    adi_nat_rank +                 
    newborn_race +                
    nicu_indication +             
    newborn_los_cat + 
    newborn_los +
    insurance_group                
  | group_pooled,
  data = meta_omics,
  render.continuous = c("Median [Q1, Q3]" = "MEDIAN [Q1, Q3]"),
  extra.col = list(`P value` = pvalue),
  overall = FALSE
)

table_one_omics_pooled

