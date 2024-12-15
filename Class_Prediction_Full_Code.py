from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from keras.src.layers.pooling.max_pooling1d import MaxPooling1D
from keras.layers import LSTM, Dense, Flatten, Dropout, Input
from keras.src.layers.convolutional.conv1d import Conv1D
from keras.src.layers.core.embedding import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import ast

warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation.*")

# Convert string representation of a list to an actual list using ast.literal_eval
def listing(x):
    try:
        return ast.literal_eval(x)
    except:
        return x
    
def label_seperation(label, data):
    specified_label = data[data['Label'].apply(lambda x: True if label in x else False)]
    return specified_label

def label_seperation_binary(label, data):
    specified_label = data['Label'].apply(lambda x: 1 if label in x else 0)
    return specified_label

def padded_data(skills_list):
    padded_skills = sequence.pad_sequences(skills_list, maxlen=55)
    return pd.Series([list(arr) for arr in padded_skills])

def calculate_metrics(y_true, y_pred, threshold=0.5):
    # Binarize the predictions based on the threshold
    y_pred_bin = (y_pred > threshold).astype(int)
    
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)
    return precision, recall, f1

def label_seperation_model(label, input_data):
    specified_label = input_data.apply(lambda x: 1 if label in x else 0)
    return specified_label

def AUC_ROC_plot(DNN, Test_Y, pred):
    ref = [0 for _ in range(len(Test_Y))]
    pred = (pred > 0.5).astype(int)
    ref_auc = roc_auc_score(Test_Y, ref)
    lr_auc = roc_auc_score(Test_Y, pred)
    print("ref_auc:", ref_auc)
    print("lr_auc:", lr_auc)
    ns_fpr, ns_tpr, _ = roc_curve(Test_Y, ref)
    lr_fpr, lr_tpr, _ = roc_curve(Test_Y, pred)

    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='AUC = {}'.format(round(roc_auc_score(Test_Y, pred)*100,2)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'{DNN}_{uq_name}_AUC_ROC.png', bbox_inches='tight')
    plt.show()

# # Building the CNN Model
def DNN_classify(DNN, X_train, X_test, y_train, y_test):

    model = Sequential()      
    #Embedding
    model.add(Embedding(15101, 256, input_length=max_words))#15101 is the total number of skills in the input data

    if DNN == 'LSTM':
        model.add(LSTM(32, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fitting the data onto model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=16, verbose=1)

    elif DNN == 'CNN':
        model.add(Conv1D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fitting the data onto model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)
    
    # Getting score metrics from our model
    scores = model.evaluate(X_test, y_test, verbose=0)
    # Displays the accuracy of correct sentiment prediction over test data
    print("Accuracy: %.2f%%" % (scores[1]*100))
    predicted_labels = model.predict(X_test)
    predicted_labels = predicted_labels.flatten()

    precision, recall, f1 = calculate_metrics(y_test, predicted_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Confusion Matrix
    predicted_labels = (predicted_labels > 0.5).astype(int)
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    if DNN == 'LSTM':
        disp.plot(cmap=plt.cm.Oranges)
    elif DNN == 'CNN':
        disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{DNN} Confusion Matrix for {uq_name}')
    plt.savefig(f'{DNN}_{uq_name}_Confusion_Matrix.png', bbox_inches='tight')
    plt.show()
    
    # AUC_ROC plot
    AUC_ROC_plot(DNN, y_test, predicted_labels)
    return predicted_labels, f1, model

# Read Resume Data
skill_per_resume = pd.read_csv("skill per resume importance sorted numeric.csv")
skill_per_resume['skills from resume'] = skill_per_resume['skills from resume'].apply(listing)
skill_per_resume['Label'] = skill_per_resume['Label'].apply(listing)
uniq_labels = ['Database_Administrator', 'Front_End_Developer', 'Java_Developer', 'Network_Administrator', 'Project_manager', 'Python_Developer', 'Security_Analyst', 'Software_Developer', 'Systems_Administrator', 'Web_Developer']

## Upsampling Method ##
# Get a total density of labels inside dataset
density = pd.DataFrame(columns=['label', 'count'])
for index, uq_name in enumerate(uniq_labels):
    density.at[index, 'label'] = uq_name
    density.at[index, 'count'] = len(label_seperation(uq_name, skill_per_resume))

density['count']=density['count'].apply(lambda x: x/len(skill_per_resume))
density_fixed = density.copy()

smote = SMOTE(random_state=42)
# Seperate each resume based on their labels while keeping the density of 0 data and apply upsampling on them
for uq_name in tqdm(uniq_labels, total=10, colour='navy', desc="Balancing Label Density"):
    # Seperating data based on their label as 1 labeled
    label_1 = label_seperation(uq_name, skill_per_resume)

    # To prevent 'Software_Developer' label cause an error
    if len(label_1)> 13000:
        label_1 = label_1.sample(10000)
    # Get the same amount of random resume data withscaled density of og dataset
    density['count'] = density_fixed['count']
    density['count'] = density['count'].apply(lambda x: x * len(label_1)/2)
    remaining_data = skill_per_resume[~skill_per_resume.isin(label_1.to_dict('list')).all(axis=1)]
    label_0 = pd.DataFrame()
    
    # For each label other than the target, get a scaled density number of that label's resumes
    for index, row in density.iterrows():
        if row['label'] != uq_name:
            LabelBasedData = label_seperation(row['label'], remaining_data)
            LabelBasedData = LabelBasedData.sample(int(row['count']))
            label_0 = pd.concat([label_0, LabelBasedData],axis = 0, ignore_index=False)
    label_0 = label_0.drop_duplicates(subset=['index'])
    
    # Concat the resumes into a single dataframe and mix their positions
    new_data = pd.concat([label_1, label_0],axis = 0, ignore_index=True)
    new_data = new_data.sample(len(new_data))
    new_data['Label'] = new_data['Label'].apply(listing)

    # Add binary labeling for DNN
    new_data_y = label_seperation_binary(uq_name, new_data).to_list()
    
    # Upsample the newly gathered data
    var_X_name = f"{uq_name}_balanced_X_data"
    var_y_name = f"{uq_name}_balanced_y_data"
    globals()[var_X_name], globals()[var_y_name] = smote.fit_resample(padded_data(new_data['skills from resume']).to_list(), new_data_y)

    var_name = f"{uq_name}_balanced_data" 
    globals()[var_name] = new_data.copy()
    
    print(f"{uq_name} ---> {globals()[var_y_name].count(1)} 1's && {globals()[var_y_name].count(0)} 0s now!")
print(">>Upsampling is Done!<<")


### Label Classification ###
f1_results = []
uniq_labels.sort(reverse=True)

# Select Deep Learning Method!
DNN_type = 'LSTM'

# For each resume label create a binary DNN model and evaluate their results
for uq_name in tqdm(uniq_labels, total=10, colour='lime', desc=f"Running {DNN_type} Model"):
    var_X_name = f"{uq_name}_balanced_X_data"
    var_y_name = f"{uq_name}_balanced_y_data"
    
    X_data = np.array(globals()[var_X_name])
    y_data = np.array(globals()[var_y_name])

    data_split = int(len(X_data) * 20/100)
    X_train, y_train = X_data[data_split: ], y_data[data_split: ]
    X_test, y_test = X_data[ :data_split], y_data[ :data_split]
    
    max_words = 55 
    print("\n",uq_name)
    var_name = f"{uq_name}_{DNN_type}"
    globals()[var_name], f1score, globals()[f"{uq_name}_{DNN_type}_model"] = DNN_classify(DNN_type, X_train, X_test, y_train, y_test)
    f1_results.append(f1score)
    print("-"*128)

print("Average F1 score of all labels ===> ", f"{np.array(f1_results).mean()*100:.2f}%")
