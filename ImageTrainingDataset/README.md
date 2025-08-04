# ImageTrainingDataset
Hood Canal, Puget Sound, Washington Zooplankton Image Training Dataset

# Image Training Dataset:
Below is the link to our Image Training Dataset (ImageTrainingDataset)

https://drive.google.com/file/d/15TImtLOS9ND_Vs01NzLf95iROwmTVOXd/view?usp=drive_link

Note: This dataset needs to undergo pre-processing steps (ie. image augmentation and balancing) before it should be used to train an algorithm for best results.

# Image File Naming Convention: 

File Name Example:  
SPC-UW-1534792542343244-337276376507-004392-154-2546-848-756-780.png  

 
SPC: Scripps Plankton Camera  
UW: University of Washington  
1534792542343244:  Unix Date/Timestamp [To convert to UTC, use (X / 86400) + 25569.]  
337276376507-004392: Directory and subdirectory extension specific to our system.  
154-2546-848-756-780: ROI details (x, y, w, h, image size)  

# Category Descriptions & List (ImageTrainingDataset):  

Non-intuitive:   
ClusteredSnow => Non-symmetrical "aggregated" clusters of marine snow  
Filament_Filaments => singular and/or multiple, slender filament/threadlike objects  
MarineSnow => Non-symmetrical “scattered” snow (smaller and numerous vs ClusteredSnow)  
Unknown => Images that are too blurry to confidently classify or are “unknown”  

Gelatinous:   
Anthomedusae - (Euphysa tentaculata)  
Cydippida - (Euplokamis dunlapae)  
Cydippida - (Pleurobrachia bachei)  
Cydippida - (Unknown)  
Lobata  
Siphonophore - Calycophorae (Muggiaea atlantica)  
Trachymedusae - (Aglantha digitale)  
Trachymedusae - (Pantachogon haeckeli)  
Trachymedusae - (young)  

Copepoda:    
Calanoida - (Acartia spp.)  
Calanoida - (Calanus sp.)  
Calanoida - (Centropages abdominalis)  
Calanoida - (Metridia spp.)  
Calanoida - (Psudo Micro Para) => includes pseudocalanus, microcalanus, and paracalanus  
Copepoda - (nauplii)  
Cyclopoida - (Oithona spp.)  
Harpacticoida - (Microsetella rosea)  
Poecilostomatoida - (Ditrichocoryceaus anglicus)  
Poecilostomatoida - (Triconia spp.)  

Amphipoda:    
Cyphocaridae - (Cyphocaris challengeri)  
Hyperiidea - (Themisto pacifica _ Hyperoche sp.)  
Gammeridea - (possibly Calliopius sp)  

Phytoplankton:    
Dinoflagellata - (Noctiluca)  
(Diatoms)   
  
Other:   
Chaetognatha  
Decapoda - Caridea (Shrimp)  
Euphausiacea - Euphausiidae (Krill)  
Ostracoda - (Halocyprididae)  
Fish_larvae  
Larvacea - (Oikopleura dioica)  
Pteropoda - (Clione limacina)  
Pteropoda - (Limacina helicina)  

# List of “class_names” for use in Image_Classification.py (line 90)   

['Anthomedusae - (Euphysa tentaculata)', 'Calanoida - (Acartia spp.)', 'Calanoida - (Calanus sp.)', 'Calanoida - (Centropages abdominalis)', 'Calanoida - (Metridia spp.)', 'Calanoida - (Psudo Micro Para)', 'Chaetognatha', 'ClusteredSnow', 'Copepoda - (nauplii)', 'Cyclopoida - (Oithona spp.)', 'Cydippida - (Euplokamis dunlapae)', 'Cydippida - (Pleurobrachia bachei)', 'Cydippida - (Unknown)', 'Cyphocaridae - (Cyphocaris challengeri)', 'Decapoda - Caridea (Shrimp)', 'Diatoms', 'Dinoflagellata - (Noctiluca)', 'Eggs', 'Euphausiacea - Euphausiidae (Krill)', 'Filament_Filaments', 'Fish_larvae', 'Gammeridea- (possibly Calliopius sp)', 'Harpacticoida - (Microsetella rosea)', 'Hyperiidea - (Themisto pacifica _ Hyperoche sp.)', 'Larvacea - (Oikopleura dioica)', 'Lobata', 'MarineSnow', 'Ostracoda - (Halocyprididae)', 'Poecilostomatoida - (Ditrichocoryceaus anglicus)', 'Poecilostomatoida - (Triconia spp.)', 'Pteropoda - (Clione limacina)', 'Pteropoda - (Limacina helicina)', 'Siphonophore - Calycophorae (Muggiaea atlantica)', 'Trachymedusae - (Aglantha digitale)', 'Trachymedusae - (Pantachogon haeckeli)', 'Trachymedusae - (young)', 'Unknown']
