# t1modeler data set preparation
Source code for t1modeler.com datasets preparation

**The scripts in this repository faciliate the following tasks | 本代码仓库中的脚本完成以下任务:**

1. download data file from one of the various web pages | 从各种不同的数据页面中下载原始文件
2. convert the data into pandas dataframe and binarize the target variable | 将文件中的数据转换为 pandas 数据集并创建目标变量
3. save the dataframe as CSV file which is ready for modeling on t1modeler.com | 将数据集保存为 CSV 文件，压缩后可上传至 t1modeler.com 进行模型开发

**Find the source page for each script in the table below | 表格内容为脚本与数据页面的对应关系**

| #  | File Name | Source Page |
|----|-----------|-------------|
| 1  | keel_001_kdd_cup_1999.py | [Link](https://sci2s.ugr.es/keel/dataset.php?cod=196) |
| 2  | keel_002_sonar_mines_vs_rocks.py | [Link](https://sci2s.ugr.es/keel/dataset.php?cod=85) |
| 3  | keel_003_molecular_biology.py | [Link](https://sci2s.ugr.es/keel/dataset.php?cod=181) |
| 4  | keel_004_connect_4.py | [Link](https://sci2s.ugr.es/keel/dataset.php?cod=193) |
| 5  | uci_001_adult_data_set.py | [Link](https://archive.ics.uci.edu/ml/datasets/Adult) |
| 6  | uci_002_bank_marketing.py | [Link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| 7  | uci_003_human_activity_recognition.py | [Link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) |
| 8  | uci_004_credit_approval.py | [Link](https://archive.ics.uci.edu/ml/datasets/Credit+Approval) |
| 9  | uci_005_cylinder_bands.py | [Link](https://archive.ics.uci.edu/ml/datasets/Cylinder+Bands) |
| 10 | uci_006_internet_advertisements.py | [Link](https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements) |
| 11 | uci_007_ionosphere.py | [Link](https://archive.ics.uci.edu/ml/datasets/Ionosphere) |
| 12 | uci_008_letter_recognition.py | [Link](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) |
| 13 | uci_009_multiple_features.py | [Link](https://archive.ics.uci.edu/ml/datasets/Multiple+Features) |
| 14 | uci_010_mushroom.py | [Link](https://archive.ics.uci.edu/ml/datasets/Mushroom) |
| 15 | uci_011_spambase.py | [Link](https://archive.ics.uci.edu/ml/datasets/Spambase) |
| 16 | uci_012_insurance_company_benchmark.py | [Link](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29) |
| 17 | uci_013_german_credit_data.py | [Link](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29) |
| 18 | uci_014_secom.py | [Link](https://archive.ics.uci.edu/ml/datasets/SECOM) |
| 19 | uci_015_qsar_biodegradation.py | [Link](https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation) |
| 20 | uci_016_seismic_bumps.py | [Link](https://archive.ics.uci.edu/ml/datasets/seismic-bumps) |
| 21 | uci_017_thoracic_surgery_data.py | [Link](https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data) |
| 22 | uci_018_phishing_websites.py | [Link](https://archive.ics.uci.edu/ml/datasets/Phishing+Websites) |
| 23 | uci_019_default_of_credit_card_clients.py | [Link](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) |
| 24 | uci_020_sports_articles_objectivity.py | [Link](https://archive.ics.uci.edu/ml/datasets/Sports+articles+for+objectivity+analysis) |
| 25 | uci_021_heart_disease.py | [Link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| 26 | uci_022_dermatology.py | [Link](https://archive.ics.uci.edu/ml/datasets/Dermatology) |
| 27 | uci_023_madelon.py | [Link](https://archive.ics.uci.edu/ml/datasets/Madelon) |
| 28 | uci_024_ozone_level_detection.py | [Link](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection) |
| 29 | uci_025_parkinsons.py | [Link](https://archive.ics.uci.edu/ml/datasets/Parkinsons) |
| 30 | uci_026_cardiotocography.py | [Link](https://archive.ics.uci.edu/ml/datasets/Cardiotocography) |
| 31 | uci_027_miniboone_particle_identification.py | [Link](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification) |
| 32 | uci_028_gas_sensor_array_drift.py | [Link](https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset) |
| 33 | uci_029_cnae_9.py | [Link](https://archive.ics.uci.edu/ml/datasets/CNAE-9) |
| 34 | uci_030_climate_model_simulation_crashes.py | [Link](https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes) |
| 35 | uci_031_eeg_eye_state.py | [Link](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State) |
| 36 | uci_032_lsvt_voice_rehabilitation.py | [Link](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) |
| 37 | uci_033_urban_land_cover.py | [Link](https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover) |
| 38 | uci_034_diabetes_130_us_hospitals.py | [Link](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) |
| 39 | uci_035_gesture_phase_segmentation.py | [Link](https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation) |
| 40 | uci_036_student_performance.py | [Link](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |
| 41 | uci_037_sensorless_drive_diagnosis.py | [Link](https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis) |
| 42 | uci_038_tv_news_channel_commercial_detection.py | [Link](https://archive.ics.uci.edu/ml/datasets/TV+News+Channel+Commercial+Detection+Dataset) |
| 43 | uci_039_diabetic_retinopathy_debrecen.py | [Link](https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set) |
| 44 | uci_040_online_news_popularity.py | [Link](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) |
| 45 | uci_041_mice_protein_expression.py | [Link](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) |
| 46 | uci_042_occupancy_detection.py | [Link](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+) |
| 47 | uci_043_gas_sensors_for_home_activity.py | [Link](https://archive.ics.uci.edu/ml/datasets/Gas+sensors+for+home+activity+monitoring) |
| 48 | uci_044_polish_companies_bankruptcy.py | [Link](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data) |
| 49 | uci_045_htru2.py | [Link](https://archive.ics.uci.edu/ml/datasets/HTRU2) |
| 50 | uci_046_cervical_cancer.py | [Link](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29) |
| 51 | uci_047_epileptic_seizure_recognition.py | [Link](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) |
| 52 | uci_048_burst_header_packet.py | [Link](https://archive.ics.uci.edu/ml/datasets/Burst+Header+Packet+%28BHP%29+flooding+attack+on+Optical+Burst+Switching+%28OBS%29+Network) |
| 53 | uci_049_extention_of_z_alizadeh_sani.py | [Link](https://archive.ics.uci.edu/ml/datasets/extention+of+Z-Alizadeh+sani+dataset) |
| 54 | uci_050_ida2016challenge.py | [Link](https://archive.ics.uci.edu/ml/datasets/IDA2016Challenge) |
| 55 | uci_051_hcc_survival.py | [Link](https://archive.ics.uci.edu/ml/datasets/HCC+Survival) |
| 56 | uci_052_online_shoppers_purchasing_intention.py | [Link](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) |
| 57 | uci_053_electrical_grid_stability.py | [Link](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+) |
| 58 | uci_054_caesarian_section_classification.py | [Link](https://archive.ics.uci.edu/ml/datasets/Caesarian+Section+Classification+Dataset) |
| 59 | uci_055_audit_data.py | [Link](https://archive.ics.uci.edu/ml/datasets/Audit+Data) |
| 60 | uci_056_hepatitis_c_virus.py | [Link](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients) |
| 61 | uci_057_glass_identification.py | [Link](https://archive.ics.uci.edu/ml/datasets/glass+identification) |
| 62 | uci_058_iris.py | [Link](https://archive.ics.uci.edu/ml/datasets/iris) |
| 63 | uci_059_optical_recognition_of_handwritten_digits.py | [Link](http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits) |
| 64 | vanderbilt_001_titanic.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.html) |
| 65 | vanderbilt_002_acute_bacterial_meningitis.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/abm.html) |
| 66 | vanderbilt_003_ari_dataset.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/bin/view/Main/AriDescription) |
| 67 | vanderbilt_004_duchenne_muscular_dystrophy.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/dmd.html) |
| 68 | vanderbilt_005_right_heart_catheterization.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/rhc.html) |
| 69 | vanderbilt_006_ucla_stress_echocardiography.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/stressEcho.html) |
| 70 | vanderbilt_007_support_study.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc) |
| 71 | vanderbilt_008_very_low_birth_weight_infants.py | [Link](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/vlbw.html) |
