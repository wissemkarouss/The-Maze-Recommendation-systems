
#ran on Kaggle
!pip install sentence-transformers
!pip install torch
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm  # For tracking progress in batches

# check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load dataset
dataset = pd.read_csv('/kaggle/input/d/infamouscoder/dataset-netflix-shows/netflix_titles.csv')

# load model to GPU if available
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# combine fields (title, genre, description) for embeddings
def combine_description_title_and_genre(description, listed_in, title):
    return f"{description} Genre: {listed_in} Title: {title}"

# create combined text column
dataset['combined_text'] = dataset.apply(lambda row: combine_description_title_and_genre(row['description'], row['listed_in'], row['title']), axis=1)

# generate embeddings in batches to save memory
batch_size = 32
embeddings = []

for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Embeddings"):
    batch_texts = dataset['combined_text'][i:i+batch_size].tolist()
    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
    embeddings.extend(batch_embeddings.cpu().numpy())  # move to CPU to save memory

# convert list to numpy array
embeddings = np.array(embeddings)

# save embeddings and metadata
np.save("/kaggle/working/netflix_embeddings.npy", embeddings)
dataset[['show_id', 'title', 'description', 'listed_in']].to_csv("/kaggle/working/netflix_metadata.csv", index=False)
