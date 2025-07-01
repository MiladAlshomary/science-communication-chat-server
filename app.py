import streamlit as st
import random
import time
import openai
import pdfplumber
import tiktoken
import re
import json
from datetime import datetime
import os
import base64

# Configure OpenAI API key
# Create a secrets.toml file in your .streamlit folder:
# OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
journalist_prompt = st.secrets.get("JOURNALIST_PROMPT")
baseline_journalist_prompt = st.secrets.get("BASELINE_JOURNALIST_PROMPT")

MODEL_CONFIGS = {
    "GPT-3.5 Turbo": {
        "model": "gpt-3.5-turbo",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
        "prompt": baseline_journalist_prompt,

    },
    "Baseline Llama": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_url": "http://localhost:7777/v1",
        "api_key": "not-needed",
        "prompt": baseline_journalist_prompt,

    },
    "Fine-tuned Llama": {
        "model": "llm_journalist",
        "base_url": "http://localhost:7777/v1",
        "api_key": "not-needed",
        "prompt": journalist_prompt,

    },
    "Baseline Qwen": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_url": "http://localhost:7000/v1",
        "api_key": "not-needed",
        "prompt": baseline_journalist_prompt,

    },
    "Fine-tuned Qwen": {
        "model": "llm_journalist",
        "base_url": "http://localhost:7000/v1",
        "api_key": "not-needed",
        "prompt": journalist_prompt,

    },
}

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit secrets. GPT models will not work.", icon="⚠️")

def save_session_to_json():
    """Saves the current chat session to a JSON file."""
    if not st.session_state.get("last_uploaded_filename") or not st.session_state.get("messages"):
        return

    # Don't save if it's just the initial message or empty
    if len(st.session_state.messages) <= 1:
        return

    try:
        # Sanitize filename from paper title
        paper_title = st.session_state.last_uploaded_filename
        base_title = os.path.splitext(paper_title)[0]
        sanitized_title = re.sub(r'[^\w\s-]', '', base_title).strip().replace(' ', '_')

        # Get current date for filename
        current_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create filename
        filename = f"{sanitized_title}_{current_date_str}.json"

        # Prepare data for JSON
        session_data = {
            "paper_title": paper_title,
            "paper_text": st.session_state.get("extracted_introduction", ""),
            "api_config" : MODEL_CONFIGS[st.session_state.selected_model_name],
            "date": datetime.now().isoformat(),
            "messages": st.session_state.messages,
        }

        # Create a directory to store sessions if it doesn't exist
        sessions_dir = "sessions"
        if not os.path.exists(sessions_dir):
            os.makedirs(sessions_dir)

        # Save to file
        filepath = os.path.join(sessions_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False)

        st.sidebar.info(f"Previous session saved to {filepath}")
    except Exception as e:
        st.sidebar.error(f"Failed to save session: {e}")

# Function to call OpenAI API
def call_openai_api(messages_for_api, selected_model_name):
    config = MODEL_CONFIGS[selected_model_name]
    client = openai.OpenAI(base_url=config["base_url"], api_key=config["api_key"])

    completion = client.chat.completions.create(
        model=config["model"],
        messages=messages_for_api,
        temperature=0.7,
        top_p=0.9,
        extra_body={"temperature": 0.97, "top_p": 0.9, "min_new_tokens":10},
    )
    assistant_response = completion.choices[0].message.content
    return assistant_response.replace("Journalist:", "").replace("[name],", "")

# Function to extract introduction from PDF
def extract_introduction_from_pdf(pdf_file_object):
    """
    Extracts text identified as the 'Introduction' from a PDF.
    Returns the extracted text or None if not found or an error occurs.
    """
    if pdf_file_object is None:
        return None

    try:
        with pdfplumber.open(pdf_file_object) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=10)
                if page_text:
                    full_text += page_text + "\n"

            extracted_intro = full_text

            if extracted_intro:
                try:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    tokens = encoding.encode(extracted_intro)
                    max_tokens = 1500
                    if len(tokens) > max_tokens:
                        truncated_tokens = tokens[:max_tokens]
                        extracted_intro = encoding.decode(truncated_tokens) + "..."
                except Exception as e:
                    st.sidebar.warning(f"Could not tokenize/truncate introduction: {e}")
            return extracted_intro if extracted_intro else None

    except Exception as e:
        st.sidebar.error(f"Error processing PDF: {e}")
        return None


def loading_intro_from_pdf():
    # Sidebar for model selection
    st.sidebar.title("🤖 LLM Selection")
    llm_options = list(MODEL_CONFIGS.keys())

    # Initialize model in session state if not present
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = llm_options[0]

    # Get index of the currently selected model
    try:
        current_index = llm_options.index(st.session_state.selected_model_name)
    except ValueError:
        current_index = 0

    # Create the selectbox to get user's choice
    selected_model_name = st.sidebar.selectbox(
        "Choose a model to chat with:",
        options=llm_options,
        index=current_index,
    )
    # If the model is changed, save the session, clear the chat, and rerun
    if selected_model_name != st.session_state.selected_model_name:
        save_session_to_json()
        st.session_state.selected_model_name = selected_model_name
        st.session_state.messages = []
        st.rerun()

    # Sidebar for PDF upload
    st.sidebar.title("📄 Paper Context")
    uploaded_file = st.sidebar.file_uploader("Upload your paper here", type="pdf")

    # Manage extracted introduction in session state
    if "extracted_introduction" not in st.session_state:
        st.session_state.extracted_introduction = None
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if "pdf_file_bytes" not in st.session_state:
        st.session_state.pdf_file_bytes = None

    if uploaded_file is not None:
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            # Save the previous session before starting a new one
            save_session_to_json()

            with st.spinner(f"Processing {uploaded_file.name}..."):
                st.session_state.extracted_introduction = extract_introduction_from_pdf(uploaded_file)
                if st.session_state.extracted_introduction:
                    st.session_state.pdf_file_bytes = uploaded_file.getvalue()
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    st.sidebar.success("Introduction extracted successfully!")
                    # Clear message history to start a new session for the new PDF
                    st.session_state.messages = []
                else:
                    # Failed to extract from new PDF, keep old context if any, don't change filename tracking
                    st.sidebar.warning(f"Could not extract 'Introduction' from '{uploaded_file.name}' or PDF is unreadable/empty. Previous context (if any) is retained.")
        
        if st.session_state.get("pdf_file_bytes"):
            expander_title = "View Uploaded PDF"
            if st.session_state.last_uploaded_filename:
                expander_title += f" ({st.session_state.last_uploaded_filename})"
            with st.sidebar.expander(expander_title, expanded=True):
                base64_pdf = base64.b64encode(st.session_state.pdf_file_bytes).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
    elif st.session_state.last_uploaded_filename is not None: # File was removed by user
        # Also save session when file is removed
        save_session_to_json()
        st.session_state.extracted_introduction = None
        st.session_state.last_uploaded_filename = None
        st.session_state.pdf_file_bytes = None
        st.session_state.messages = [{"role": "assistant", "content": "Please upload a paper from the sidebar to start chatting."}]
        st.sidebar.info("PDF context removed.")
        # Rerun to update the chat display immediately
        st.rerun()

st.write("LLM Science Explainers!")

loading_intro_from_pdf()

# Initialize chat history only if it doesn't exist (e.g., first app run).
# If a PDF is uploaded, messages might be cleared to [], so this won't re-initialize then.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please upload a paper from the sidebar to start chatting."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input only if a paper has been uploaded
if st.session_state.get("extracted_introduction"):
    if prompt := st.chat_input("Type your answer here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if st.session_state.get("selected_model_name"):
                try:
                    # Get config for selected model
                    selected_model_key = st.session_state.selected_model_name
                    config = MODEL_CONFIGS[selected_model_key]
                    api_messages_for_openai = []

                    api_messages_for_openai.append({
                        "role": "system",
                        "content": config["prompt"]
                    })
                    # Add PDF introduction as a system message if available
                    if st.session_state.extracted_introduction:
                        api_messages_for_openai.append({
                            "role": "user",
                            "content": "[PAPER]\n{}".format(st.session_state.extracted_introduction)
                        })
                    
                    # Add the rest of the conversation history
                    # st.session_state.messages already includes the latest user prompt
                    for m in st.session_state.messages:
                        api_messages_for_openai.append({"role": m["role"], "content": m["content"]}) # This includes the latest user prompt

                    print(f"Calling API for user prompt with model: {selected_model_key}")
                    assistant_response = call_openai_api(api_messages_for_openai, selected_model_key)
                except Exception as e:
                    st.error(f"Error communicating with AI service: {e}")
                    assistant_response = "Sorry, I encountered an error with the AI service."
            else:
                assistant_response = "Please select a model from the sidebar."

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    # Show a disabled chat input if no paper is uploaded
    st.chat_input("Upload a paper to start chatting", disabled=True)

# If a PDF was uploaded and the chat is empty (new session for this PDF), trigger a default query.
if st.session_state.extracted_introduction and not st.session_state.messages:
    with st.chat_message("user"):
        st.markdown("Upload your paper, and lets start chatting!")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.get("selected_model_name"):
            try:
                # Get config for selected model
                selected_model_key = st.session_state.selected_model_name
                config = MODEL_CONFIGS[selected_model_key]

                # Create client on the fly
                client = openai.OpenAI(base_url=config["base_url"], api_key=config["api_key"])
                # Prepare messages for OpenAI API, including the introduction and the default question.
                api_messages_for_openai = []

                api_messages_for_openai.append({"role": "assistant", "content": config["prompt"]})

                if st.session_state.extracted_introduction:
                    api_messages_for_openai.append({"role": "user", 
                                                    "content": "[PAPERT-TITLE]\n{}\n[PAPER]\n{}".format(st.session_state.last_uploaded_filename, st.session_state.extracted_introduction)})

                print(config)
                completion = client.chat.completions.create(
                    model=config["model"],
                    messages=api_messages_for_openai,
                    temperature=0.7,
                    top_p=0.9,
                )
                assistant_response = completion.choices[0].message.content
                assistant_response = assistant_response.replace("Journalist:", "")
                assistant_response = assistant_response.replace("[name],", "")
            except Exception as e:
                print(e)
                st.error(f"Error communicating with AI service: {e}")
                assistant_response = "Sorry, I encountered an error with the AI service."
        else:
            assistant_response = "Please select a model from the sidebar."

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Update chat history with the assistant's response to the default query.
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Rerun to display the first message and wait for user input
    st.rerun()
