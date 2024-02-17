import streamlit as st
from src.data_ingestion import load_data
from src.embedding import download_gemini_embedding
from src.model_api import load_model
import os
from logger import logging
    
def main():
    st.set_page_config("QA with Documents")
    
    doc=st.file_uploader("upload your document")
    
    st.header("QA with Documents(Information Retrieval)")
    
    user_question= st.text_input("Ask your question")
    
    if st.button("submit & process"):
        with st.spinner("Processing..."):
            save_path = os.path.join("data", "doc")
            
            with open(save_path, "wb") as f:
                f.write(doc.getbuffer())
            
            logging.info("saved orginal file")

            document=load_data(save_path)
            model=load_model()
            query_engine=download_gemini_embedding(model,document)
                
            response = query_engine.query(user_question)
                
            st.write(response.response)
                
                
if __name__=="__main__":
    main()          
                
    
    
    
    
    