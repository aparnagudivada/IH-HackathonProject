from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
import dotenv
import getpass
import time 
import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

# Load environment variables
load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Example: setting these variables in the environment
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Define the directory and path for the FAISS index
faiss_dir = "faiss_index_dir"
os.makedirs(faiss_dir, exist_ok=True)
faiss_index_path = os.path.join(faiss_dir, "faiss_index")

# Check if the FAISS index already exists
if os.path.exists(faiss_index_path):
    # Load the FAISS index from disk
    db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    # Step 1: Load documents from CSV
    raw_documents = CSVLoader("drugs.csv", encoding="utf-8").load()

    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
    )
    documents = text_splitter.split_documents(raw_documents)

    # Step 3: Create embeddings and FAISS index
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings_model)

    # Save the FAISS index to disk
    db.save_local(faiss_index_path)

# Step 4: Create the retriever
retriever = db.as_retriever()

# Step 5: Set up the language model and QA chain
llm_src = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
qa_chain = create_qa_with_sources_chain(llm_src)
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)


# Replace with the actual import statement for your retrieval_qa function

# Define the chatbot logic as a function
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to append messages to the chat history
def append_to_chat_history(role, content):
    st.session_state.messages.append({"role": role, "content": content})
def prepare_history():
    history= [
        f"{message['role']}:{message['content']}"
        for message in st.session_state.messages
    ]
    chat_history= [(message.split(":")[0],message.split(":")[1]) for message in history]
    print(chat_history)
    return chat_history

def chatbot():
    st.markdown(
        """
        <style>
            body {
                background-color: #110159; /* Light blue background */
            }
            .stApp {
                background-color: #110159; /* Ensure the app background is also light blue */
            }
            .stTextInput>div>input {
            color: white; /* Set text color to white */
            background-color: #333; /* Optional: set background color for better contrast */
            border: 1px solid #555; /* Optional: border color for the input box */

        }
        .st-emotion-cache-1whx7iy p {
            color:white;
            }
        .stMarkdown, .stWrite,.stTextInput, .stChatMessage {
            color: white; /* Set text color for st.write and chat messages */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
     # Apply custom CSS
    apply_custom_css()
    logo_path = "Logo.png"
    logo_image = Image.open(logo_path)
    # st.markdown('<div class="title">MediBot</div>', unsafe_allow_html=True)
    # st.markdown("<p style='color: #ff5d55;'><i>Informing You to Live Your Healthiest Life</i></p>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{image_to_base64(logo_image)}" class="logo" alt="Logo">
            <div class="title">MediBot</div>
        </div>
        <p class="custom-text">Informing You to Live Your Healthiest Life</p>
        """,
        unsafe_allow_html=True
    )
    

    # Step 1: Ask for medications
    with st.chat_message(name="assistant"):
        medications = st.text_input("What medications are you using?")
    

    

    if medications:
        # Step 2: Check if medications are in the list
        result = retrieval_qa({
            "question": f"Are the following medications in the list? {medications}.If the medication is not list start your answer with No",
             "chat_history": []
        })

        if(result['answer'].startswith('No')):
            st.write("I am not able to find the information related to "+medications)
            return
        result1 = retrieval_qa({
            "question": f" What are the side effects of using {medications}? Include common side effects, serious side effects?",
            "chat_history": []
        })
        with st.chat_message(name="assistant"):
            st.write( result1['answer'])
        append_to_chat_history("Bot", result1['answer'] )
        # Step 3: Ask for symptoms
        time.sleep(1)
        with st.chat_message(name="assistant"):
            symptoms = st.text_input("What are the symptoms you are facing?")
        if symptoms:
            # Check if symptoms are related to the medications
            print(medications)
            result = retrieval_qa({
                "question": f" Just Tell me if the {symptoms} is a side effect of {medications}? if it is not a side effect start your answer with No",
                 "chat_history": prepare_history()
            })
            with st.chat_message(name="assistant"):
                st.write(result['answer'])
            append_to_chat_history("Bot", result['answer'])
            if(result['answer'].startswith('No')):
                with st.chat_message(name="assistant"):
                    st.write('Your symptoms may not be related to  your medications.Please consult your doctor to know the more about it')
                    return

             # # Step 4: Ask about the history of symptoms
            time.sleep(1)
            with st.chat_message(name="assistant"):
                history = st.text_input("Have you faced these symptoms before starting your current medication?")

            if history == "Yes" or history  == "yes":
                with st.chat_message(name="assistant"):
                    st.write("Hmm, it looks like your symptoms might not be related to the medications.Please consult a doctor for more information")
                    return
            elif history== "No" or history == "no":
                time.sleep(1)
                with st.chat_message(name="assistant"):
                    duration = st.text_input("How long has this been happening?")
                if(duration):
                    time.sleep(1)
                    with st.chat_message(name="assistant"):
                        severity = st.text_input("How severe are your symptoms? (High, Low)")
                    if severity.lower() == "high":
                        time.sleep(1)
                        with st.chat_message(name="assistant"):
                            st.write("Please go and consult your doctor for more advice!!")
                            st.write("If you have any more questions or concerns in the future, don't hesitate to ask. Take Care and Stay Healthy!!")

                    elif severity.lower() == "low":
            #         # Provide temporary relieving solutions
                        result = retrieval_qa({
                        "question": f"Give 2 temporary relieving solutions to the {symptoms} provided",
                        "chat_history": []
                    })
                        with st.chat_message(name="assistant"):
                            st.write(result['answer'])
                            time.sleep(1)
                            st.write("If the symptoms still persist please consult a doctor")  # Give 2 or 3 simple solutions
                            st.write("If you have any more questions or concerns in the future, don't hesitate to ask. Take Care and Stay Healthy!!")

# Main function to handle login and guest access

import streamlit as st
def apply_custom_css():
    st.markdown(
        """
        <style>
        .title {
            color: white;
            font-size: 48px; /* Adjust the font size as needed */
        }
        .custom-label {
            color: white;
            font-weight: bold; /* Optional: make labels bold */
            font-size: 16px; /* Optional: adjust font size */
            
        }
        .st-emotion-cache-ue6h4q{
            min-height:0px;
        }
         .logo {
            height: 100px; /* Adjust the height of the logo */
            width: auto; /* Maintain aspect ratio */
            vertical-align: middle;
            margin-right: -17px;
            margin-top: 13px;
        }
        .header-container {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def login_page():
    st.markdown(
        """
        <style>
            body {
                background-color: #110159; /* Light blue background */
            }
            .stApp {
                background-color: #110159; /* Ensure the app background is also light blue */
            }
            .login-container {
                background-color: white; /* White background for the login box */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                width: 400px;
                margin: 0 auto;
            }
            .custom-text{
                color: #ff5d55;
                    padding-left: 83px;
                    margin-top: -25px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    apply_custom_css()  # Apply custom CSS
    logo_path = "Logo.png"
    logo_image = Image.open(logo_path)
    # st.markdown('<div class="title">MediBot</div>', unsafe_allow_html=True)
    # st.markdown("<p style='color: #ff5d55;'><i>Informing You to Live Your Healthiest Life</i></p>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{image_to_base64(logo_image)}" class="logo" alt="Logo">
            <div class="title">MediBot</div>
        </div>
        <p class="custom-text">Informing You to Live Your Healthiest Life</p>
        """,
        unsafe_allow_html=True
    )
    # Username and password input fields
    #username = st.text_input("UserName")
    #password = st.text_input("Password", type="password")

    st.markdown('<label class="custom-label">Username</label>', unsafe_allow_html=True)
    username = st.text_input("", key="username", placeholder="Enter your username")

    st.markdown('<label class="custom-label">Password</label>', unsafe_allow_html=True)
    password = st.text_input("", key="password", type="password", placeholder="Enter your password")


    # Hardcoded credentials for example purposes
    correct_username = "user"
    correct_password = "password"

    # Login button
    if st.button("Login"):
        if username == correct_username and password == correct_password:
            st.session_state.logged_in = True
            st.session_state.page = "chatbot"
            
        else:
            st.error("Invalid username or password")

    # Continue as Guest button
    if st.button("Continue as Guest"):
        st.session_state.logged_in = True
        st.session_state.page = "chatbot"
        st.rerun()
def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "chatbot":
        chatbot()

if __name__ == "__main__":
    main()
