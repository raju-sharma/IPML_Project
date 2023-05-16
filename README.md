# FIle 1: CNN_Cataract_image.ipynb
This file consist of CNN implemention for binary classification of cataract and normal images
# FIle 2: SVM_Pre_preocessing_weight_label.ipynb
This file consists of pre_processing using Sobel Operator and SVM implementation with label assigned as weights
# FIle 3: SVM_on_pixels.ipynb
This file consists of code in which SVM is directly applied on pixel after resizing 500*500
Above files also contains multi class just change
CATEGORIES = ['1_normal','2_cataract']
to
CATEGORIES = ['1_normal','2_cataract','2_glaucoma','3_retina_disease']
