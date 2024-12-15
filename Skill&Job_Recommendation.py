import pandas as pd
import numpy as np
import ast
from tqdm.notebook import tqdm
from sklearn.metrics import ndcg_score, dcg_score
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import random
import math
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import ast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Predict the labels for new data using a trained Deep Learning model
def DNN_predict_new_data(model, X_new):
    # Predict probabilities for the input data
    predicted_labels = model.predict(X_new)
    # Flatten the predictions to a 1D array
    predicted_labels = predicted_labels.flatten()
    # Convert probabilities to binary labels (0 or 1) based on a threshold of 0.5
    predicted_labels = (predicted_labels > 0.5).astype(int)
    return predicted_labels

# Converts a string representation of a list into an actual list
def listing(x):
    try:
        return ast.literal_eval(x)
    except:
        return x

# Flattens a nested list into a single-level list
def flatten_list(nested_list):
    # Recursively expand nested lists; keep single items intact
    return [item for sublist in nested_list for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

# Prepares a dictionary of skills and their importance scores from a DataFrame
def preprocess_imp(df):
    return {row['skill']: row['score'] for _, row in df.iterrows()}

# Calculates Jaccard Similarity between job skills and resume skills based on importance scores
def jaccard_similarity(job_skills, resume_tokens, dic_imp):
    job_tokens = set(job_skills)
    common = 0
    union = 0
    for skill in resume_tokens.union(job_tokens):
        imp_score = dic_imp.get(skill, 0)  # Get skill importance or default to 0
        union += imp_score
        if skill in resume_tokens and skill in job_tokens:
            common += imp_score
    # Return the weighted Jaccard similarity, avoiding division by zero
    return common / union if union != 0 else 0

# Recommends jobs based on resume skills and importance scores
def job_recommendation(Skills, Labels):
    resume_tokens = set(Skills)
    df_imp = pd.DataFrame(columns=['skill', 'score'])
    # Aggregate skill importance data for all labels
    for lbl in Labels:
        var_name = f"{lbl}_skill_importance"
        df_imp = pd.concat([df_imp, globals()[var_name]], axis=0, ignore_index=False)
    # Sort by importance and remove duplicates
    df_imp = df_imp.sort_values("score", ascending=False).drop_duplicates(subset=['skill'])
    dic_imp = preprocess_imp(df_imp)
    # Calculate similarity scores for each job
    skill_per_job['Similarity'] = skill_per_job['skills'].apply(lambda x: jaccard_similarity(x, resume_tokens, dic_imp))
    # Select top 10 jobs based on similarity
    top_jobs = skill_per_job.sort_values('Similarity', ascending=False).head(10)
    cat_list = top_jobs['Category'].tolist()
    result = pd.DataFrame({'Resume': [Labels], 'Jobs': [cat_list]})
    # Relevance scoring and evaluation
    recommendation_order = result['Jobs'].iloc[0]
    relevance_score = []
    for name_label in recommendation_order:
        try:
            indx = fields[fields['Category'] == name_label].index[0]
            if any(item in Labels for item in fields['Normalized'].loc[indx].split(',')):
                relevance_score.append(3)
            elif name_label in fields['Category'].to_list():
                relevance_score.append(2)
        except:
            relevance_score.append(0)
    relevant_score = np.fromiter(relevance_score, dtype=int).reshape(1, -1)
    relevance_score.sort(reverse=True)
    true_relevance = np.fromiter(relevance_score, dtype=int).reshape(1, -1)
    # Compute nDCG score
    dcg = dcg_score(true_relevance, relevant_score)
    idcg = dcg_score(true_relevance, true_relevance)
    ndcg = dcg / idcg if idcg != 0 else 0
    return 0 if len(Skills) == 0 else ndcg

# Generates embeddings for a dataset using BERT
def BERT_embedding(dataset, label):
    random_seed = 42  # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Load BERT tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')  # Load BERT model
    input_embeddings = []
    dataset = list(dataset)
    tokenized_text = tokenizer(dataset, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokenized_text)  # Generate embeddings
    embeddings = outputs['pooler_output']
    input_embeddings.append(embeddings)
    return input_embeddings

# Plots embeddings in a 2D space using t-SNE
def plot_embeddings(embeddings, label):
    tsne = TSNE(n_components=2, perplexity=1, random_state=42)
    embeddings = np.array(embeddings).squeeze()
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(5, 4))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', alpha=0.85)
    plt.title(f'2D projection of BERT embeddings for {label}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Finds the top 5 similar keywords based on input
def top_5(input_keyword, labels):
    result = {}
    tokenized_text = tokenizer(input_keyword, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokenized_text)
    input_keyword_embedding = outputs['pooler_output'].numpy()
    for label in labels:
        var_name = f"{label}_embeddings"
        input_embeddings_np = np.vstack([emb.numpy() for emb in globals()[var_name]])
        similarities = cosine_similarity(input_keyword_embedding, input_embeddings_np).flatten()
        with open(f'Label-based skills/{label} Label Skills.npy', 'rb') as f:
            unique_skills = np.load(f)
        top_5_indices = similarities.argsort()[-20:][::-1]
        top_5_keywords = [(unique_skills[i], similarities[i]) for i in top_5_indices]
        result[label] = top_5_keywords
    return result

# Maps a list of skills to their indices using a dictionary
def map_skill_dict(keys_list, skill_dict):
    # For each skill in the input list, return its corresponding index if it exists in the dictionary
    return [skill_dict[key] for key in keys_list if key in skill_dict]

# Reverts indices back to their corresponding skills using a dictionary
def map_rev(keys_list, rev_skill_dict):
    # For each index in the input list, return its corresponding skill if it exists in the dictionary
    return [rev_skill_dict[key] for key in keys_list if key in rev_skill_dict]

# Sorts recommendations based on label-specific skill importance
def label_based_sort(df_row):
    df_imp = pd.DataFrame(columns=['skill', 'score'])
    # Aggregate skill importance data for all labels in the row
    for lbl in df_row['predict']:
        var_name = f"{lbl}_skill_importance"
        df_imp = pd.concat([df_imp, globals()[var_name]], axis=0, ignore_index=False)
    # Sort by importance and remove duplicates
    df_imp = df_imp.sort_values("score", ascending=False).drop_duplicates(subset=['skill'])
    # Create dictionaries for mapping skills to indices and vice versa
    skill_dict = {row['skill']: index for index, row in df_imp.iterrows()}
    rev_skill_dict = {value: key for key, value in skill_dict.items()}
    # Map skills to indices, sort them, and revert back to original skills
    df_row['recommend'] = sorted(map_skill_dict(df_row['recommend'], skill_dict))
    df_row['recommend'] = map_rev(df_row['recommend'], rev_skill_dict)
    return df_row

# Retrieves the importance score of a specific skill for given labels
def importance_scores(TargetSkill, labels):
    df_imp = pd.DataFrame(columns=['skill', 'score'])
    # Aggregate skill importance data for all labels
    for lbl in labels:
        var_name = f"{lbl}_skill_importance"
        df_imp = pd.concat([df_imp, globals()[var_name]], axis=0, ignore_index=False)
    # Sort by importance and remove duplicates
    df_imp = df_imp.sort_values("score", ascending=False).drop_duplicates(subset=['skill'])
    # Retrieve and return the importance score for the target skill
    imp_score = df_imp[df_imp['skill'] == TargetSkill]['score']
    return imp_score.iloc[0] if not imp_score.empty else None

######      ######      ######      ######      ######      ######      ######        ######       ######       ######     

# Read and PreProccess Resume  Data
skill_per_resume = pd.read_csv("skill per resume importance sorted numeric.csv")
skill_per_resume['skills from resume'] = skill_per_resume['skills from resume'].apply(listing)
skill_per_resume['Label'] = skill_per_resume['Label'].apply(listing)

# Load skill importance data for each unique label
for uq_name in uniq_labels:
    var_name = f"{uq_name}_skill_importance"
    globals()[var_name] = pd.read_csv(f"Label-based Skills Importance/{var_name}.csv").sort_values("score", ascending=False)

# Take The smaple_data which is out Lowly scored resumes
sample_data = skill_per_resume.loc[LowData.sample(1000).index]
sample_data_padded = padded_data(sample_data['skills from resume']).to_list() # Padding the numeric-ordered skills to better match the DNN model

# Send sample data to the trained model from >2< to have their labels predicted
for uq_name in uniq_labels:
    var_name = f"{uq_name}_{DNN_type}_model"
    var_predict = f"{uq_name}_predicted_label"
    globals()[var_predict] = DNN_predict_new_data(globals()[var_name], np.array(sample_data_padded))

# Prepare the final_data dataframe
test_data = sample_data.copy()
test_data['predict'] = [[] for _ in range(len(test_data))]
test_data = test_data.reset_index()

# Assign predicted labels to each resume inside dataframe
for uq_name in tqdm(uniq_labels, total=10):
    var_predict = f"{uq_name}_predicted_label"
    indices_1 = np.where(globals()[var_predict] == 1)
    indices_1 = indices_1[0].tolist()
    test_data.loc[indices_1, 'predict'] = test_data.loc[indices_1, 'predict'].apply(lambda x: x + [uq_name])

# Add the predicted labels column to the final_data dataframe
selected_rows = skill_per_resume_copy.iloc[test_data['level_0']].drop(columns=['level_0'])
test_data = pd.concat([selected_rows.reset_index(), test_data['predict']], axis=1, ignore_index=False)
test_data['Label'] = test_data['Label'].apply(listing)
test_data['skills from resume'] = test_data['skills from resume'].apply(listing)


#################################################       Skill Recommendation Phase       #################################################


# Embed Skills gathered from highly recommended resumes ***need to update highly important skills in bert!!! both update job recommendation scores & and label-based importance***
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
for num, uq_name in enumerate(uniq_labels): 
    with open(f'Label-based skills/{uq_name} Label Skills.npy', 'rb') as f:
        skills = np.load(f)

    var_name = f"{uq_name}_embeddings"
    globals()[var_name] = BERT_embedding(skills, uq_name)
    print(f"{num+1}. {uq_name} skills BERT embedding is done!   ----->   {uq_name}: {len(skills)}")
    # plot_embeddings(np.array(globals()[var_name][0]), uq_name) #>>plot skill scatter if needed<<
print("\n-->All Embeddings are done<--")


# Main Skill Recommendation LOOP!
stop = 0
test_data['recommend'] = np.nan

for index, row in tqdm(test_data.iterrows(), total=len(test_data), colour='purple', desc="Recommending Skills"):
    recommended_skills = set()
    skills_list = row['skills from resume'] # original skills of a resume

    # If there were no labels predicted for the resume, just apply its original labels ---> unlikely in the new version
    if len(row['predict'])== 0:
        row['predict'] = (row['Label'])

    # for every skill in skills_list, recommend 5 skills. Then choose the best one for final recommendation ---> Metrics: Importance * 100 * Cosine Similarity Score(BERT)
    for skill in row['skills from resume']:
        candidate_df = pd.DataFrame()
        recommended = top_5(skill, row['predict']) # Contains the 5 most similiar skills to the in-loop skill

        # Check each candidate skill recommended by 'top_5' and choos the most suitable option
        for og_skills, recc_skills_score in recommended.items():
            top_5_results_df = pd.DataFrame(recc_skills_score, columns=['Keyword', 'Cosine Similarity']) # Creates a 1 cell dataframe with the recommended skills and their cosine scores
            candidate_df = pd.concat([candidate_df, top_5_results_df], axis=0, ignore_index=True) # Gathers all 5 recommended skills inside a single 5 cell dataframe
        if not candidate_df.empty:  # empty predicted labels could return empty recc_skills_score
            index_start = 0
            candidate_df['importance'] = (candidate_df['Keyword'].apply(lambda x: importance_scores(x, row['predict']))) * 10 # Multiply imp score with 100 to better scale it with cosine score
            candidate_df['score'] = candidate_df['Cosine Similarity'] * candidate_df['importance'] # Final metric for suitability of recommended skills
            candidate_df = candidate_df.sort_values('score', ascending=False)

            # Check if the suitable recommended skill is already present inside original skill list, is so, recommend the next one
            while candidate_df.iloc[index_start]['Keyword'] in skills_list and candidate_df.iloc[index_start]['Keyword'] in recommended_skills:
                index_start +=1
                if index_start == len(candidate_df):
                    continue

            # Add the most suitable recommended skill to the final list of recommended skills for the resume
            recommended_skills.add(candidate_df.iloc[index_start]['Keyword'])

    # Add the final recommended skills list for the resume
    test_data.at[index, 'recommend'] = str(list(recommended_skills))

# Just in case!
test_data['recommend'] = test_data['recommend'].apply(lambda x: listing(x))

# Apply label-based skill sorting to each resume
for index, row in tqdm(test_data.iterrows(), total=len(test_data), colour="navy", desc="Sorting Resume skills based on their skill importance "):
    test_data.loc[index]['recommend'] = label_based_sort(row)['recommend']


#################################################       Job Recommendation Phase       #################################################


# Read Jobs data
skill_per_job = pd.read_csv("concatenated jobs.csv")
skill_per_job['skills'] = skill_per_job['skills'].apply(listing)
fields = pd.read_csv("computer fields.csv")

# MAIN Job Recommendation LOOP!!
test_data['well recommend'] = np.nan
test_data['nDCG Before'] = np.nan
test_data['nDCG After'] = np.nan

# Compare base recommendation score with its skill recommended score. Remove skills from least important, if the score goes up, remove the next, if not after 5 times, keep the recommended skills
for index, row in tqdm(test_data.iterrows(), total=len(test_data), colour='aqua'):
    # Base Recommendation Score
    base_score = job_recommendation(row['skills from resume'], row['Label'])
    test_data.at[index, 'nDCG Before'] = base_score

    # Recommendation score with the added recommended skills
    combined_skills = row['skills from resume'] + row['recommend']
    score = job_recommendation(combined_skills, row['predict'])
    
    # Checking if some added skills were harmful to the score
    patience = 0
    new_combined_skills = combined_skills.copy()
    len_rec = len(row['recommend'])
    # Remove the least important skill and check if the results have been improved
    for _ in range(len_rec):
        combined_skills.pop(-1)
        new_score = job_recommendation(combined_skills, row['predict'])

        # if by removing a skil the results have been improved, sets the new skill list for recommendation
        if new_score > score:
            score = new_score
            new_combined_skills = combined_skills.copy()
        # if results were damaged, meaning the skill was effective to a good recommendation score but skill try 4 more times
        else:
            patience += 1
        if patience == len_rec/2: # removing recommended skills have proven damaging to the results, thus the best results are with the recommended skills intact
            break
    
    # Compare if adding recommended skills have proven effective
    if score > base_score:
        test_data.at[index, 'well recommend'] = str(list(set(new_combined_skills)))
        test_data.at[index, 'nDCG After'] = score
    else:
        test_data.at[index, 'well recommend'] = ">NRN<"  # No Recommendation Needed
        test_data.at[index, 'nDCG After'] = base_score

print("nDCG Score before Skill Recommendation:", test_data['nDCG Before'].mean())
print("nDCG Score after Skill Recommendation:", test_data['nDCG After'].mean())
test_data[['Label', 'recommend', 'well recommend', 'nDCG Before', 'nDCG After']]
