# Gradio Interface
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_input(input_type, image=None, text=None, response_amount=3):
    # initalize input variable 
    combined_input = ""

    # handle image input if chosen
    if input_type == "Image" and image:
        inputs = processor(images=image, return_tensors="pt") #process image with BlipProcessor
        out = image_model.generate(**inputs)  #generate caption with BlipModel
        image_caption = processor.decode(out[0], skip_special_tokens=True)  #decode output w/ processor
        combined_input += image_caption  # add the image caption to input

    # handle text input if chosen
    elif input_type == "Text" and text:
        combined_input += text  # add the text to input

    # handle both text and image input if chosen
    elif input_type == "Both" and image and text:
        inputs = processor(images=image, return_tensors="pt")
        out = image_model.generate(**inputs)
        image_caption = processor.decode(out[0], skip_special_tokens=True)  #repeat image processing + caption generation and decoding
        combined_input += image_caption + " and " + text  # combine image caption and text
    
    # if no input, fallback
    if not combined_input:
        combined_input = "No input provided."
    if response_amount is None:
        response_amount=3

    return vector_search(combined_input,response_amount) #search through embedded document w/ input

# load embeddings and metadata
embeddings = np.load("netflix_embeddings.npy")  #created using sentence_transformers on kaggle
metadata = pd.read_csv("netflix_metadata.csv") #created using sentence_transformers on kaggle

# vector search function
def vector_search(query,top_n=3):
    query_embedding = sentence_model.encode(query)  #encode input w/ Sentence Transformers
    similarities = cosine_similarity([query_embedding], embeddings)[0]  #similarity function
    if top_n is None:
        top_n=3
    top_indices = similarities.argsort()[-top_n:][::-1]  #return top n indices based on chosen output amount
    results = metadata.iloc[top_indices]  #get metadata
    result_text=""
    for index,row in results.iterrows():   #loop through results to get Title, Description, and Genre for top n outputs
        if index!=top_n-1:
            result_text+=f"Title: {row['title']}  Description: {row['description']}  Genre: {row['listed_in']}\n\n"
        else:
            result_text+=f"Title: {row['title']}  Description: {row['description']}  Genre: {row['listed_in']}"
    return result_text


def set_response_amount(response_amount):  #set response amount 
    if response_amount is None:
        return 3
    return response_amount

 # based on the selected input type, make the appropriate input visible
def update_inputs(input_type):
    if input_type == "Image":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    elif input_type == "Text":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    elif input_type == "Both":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
with gr.Blocks() as demo:
    gr.Markdown("# Netflix Recommendation System")
    gr.Markdown("Enter a query to receive Netflix show recommendations based on title, description, and genre.")
   
    input_type = gr.Radio(["Image", "Text", "Both"], label="Select Input Type", type="value")
    response_type=gr.Dropdown(choices=[3,5,10,25], type="value", label="Select Response Amount", visible=False)
    image_input = gr.Image(label="Upload Image", type="pil", visible=False)  # Hidden initially
    text_input = gr.Textbox(label="Enter Text Query", placeholder="Enter a description or query here", visible=False)  # hidden initially
  
    input_type.change(fn=update_inputs, inputs=input_type, outputs=[image_input, text_input, response_type])
   # state variable to store the selected response amount
    selected_response_amount = gr.State()

    # capture response amount immediately when dropdown changes
    response_type.change(fn=set_response_amount, inputs=response_type, outputs=selected_response_amount)
    
    submit_button = gr.Button("Submit")
    output = gr.Textbox(label="Recommendations")
    if selected_response_amount is None:
        selected_response_amount=3

    submit_button.click(fn=generate_input, inputs=[input_type,image_input, text_input,selected_response_amount], outputs=output)
demo.launch()
