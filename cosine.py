import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df = pd.read_csv("data/data_moods.csv")
df.info()
df.head()

col_features = df.columns[6:-3]
normalized_df= MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])

print(normalized_df[:2])

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)

def generate_recommendation(song_title, model_type=cosine ):
    """
    Purpose: Function for song recommendations 
    Inputs: song title and type of similarity model
    Output: Pandas series of recommended songs
    """
    # Get song indices
    index=indices[song_title]
    # Get list of songs for given songs
    score=list(enumerate(model_type[indices['1999']]))
    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    # Select the top-10 recommend songs
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    # Top 10 recommende songs
    top_songs=df['name'].iloc[top_songs_index]
    return top_songs
    
print("Recommended Songs:")
print(generate_recommendation('1999',cosine).values)
