import streamlit as st
import time
import openai
import fitz  # PyMuPDF
import tiktoken
import re
import json
from datetime import datetime
import os
import base64
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_pdf_viewer import pdf_viewer

# Configure OpenAI API key
# Create a secrets.toml file in your .streamlit folder:
# OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
FIREWORKAI_API_KEY = st.secrets.get("FIREWORKSAI-API-KEY")
journalist_prompt = st.secrets.get("JOURNALIST_PROMPT")
baseline_journalist_prompt = st.secrets.get("BASELINE_JOURNALIST_PROMPT")
baseline_prompt = st.secrets.get("BASELINE_PROMPT")

LAY_SUMMARY_PROMPT = """
You are an expert science communicator. Your task is to create a lay summary of a conversation between a user and a journalist AI.
The conversation is about a scientific paper.
Based on the conversation provided below, generate a concise and easy-to-understand summary for a general audience.
Focus on the key findings and their significance as discussed in the chat.
Do not include conversational filler. Present only the summary.

Here is the conversation:
"""

MODEL_CONFIGS = {
    "System 1": {
        "model": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
        "prompt": baseline_prompt,
    },
    "System 2": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_url": "http://localhost:7790/v1",
        "api_key": 'not-needed',
        "prompt": baseline_prompt,

    },
    "System 3": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_url": "http://localhost:7790/v1",
        "api_key": 'not-needed',
        "prompt": baseline_journalist_prompt,

    },
    "System 4": {
        "model": "llm_journalist",
        "base_url": "http://localhost:7777/v1",
        "api_key": "not-needed",
        "prompt": journalist_prompt,

    },
}

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit secrets. GPT models will not work.", icon="⚠️")

def _get_session_data_and_filename():
    """
    Prepares session data dictionary and a sanitized filename.
    Returns a tuple (filename, session_data_dict).
    Returns (None, None) if the session is not ready to be saved.
    """
    if not st.session_state.get("last_uploaded_filename") or not st.session_state.get("messages"):
        return None, None

    # Don't save if it's just the initial message or empty
    if len(st.session_state.messages) <= 1:
        return None, None

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
        "paper_text": st.session_state.get("extracted_introduction", ""), "user_summary": st.session_state.get("user_summary", ""),
        "api_config": st.session_state.selected_model_name,
        "date": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    }

    return filename, session_data

def save_session_to_json():
    """Saves the current chat session to a JSON file."""
    filename, session_data = _get_session_data_and_filename()
    if not filename or not session_data:
        return

    try:
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
    #print(config)
    client = openai.OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    #print(messages_for_api)
    completion = client.chat.completions.create(
        model=config["model"],
        messages=messages_for_api,
        temperature=0.7,
        top_p=0.9,
        max_completion_tokens=200,
        extra_body={"temperature": 0.3, "top_p": 0.95, "min_new_tokens":10} if False else {"temperature": 0.3, "top_p": 0.95},
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
        # Read bytes from the uploaded file object and open with fitz
        with fitz.open(stream=pdf_file_object.read(), filetype="pdf") as doc:
            full_text = ""
            for page in doc:
                page_text = page.get_text()
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
        st.session_state.conversation_summary = None
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
            st.session_state.conversation_summary = None

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
            # expander_title = "View Uploaded PDF"
            # if st.session_state.last_uploaded_filename:
            #     expander_title += f" ({st.session_state.last_uploaded_filename})"
            # with st.sidebar.expander(expander_title, expanded=True):
            #     base64_pdf = base64.b64encode(st.session_state.pdf_file_bytes).decode("utf-8")
            #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            #     st.markdown(pdf_display, unsafe_allow_html=True)
            # You can also display PDF from bytes data, for example, from a file uploader
            binary_data = uploaded_file.getvalue()

            # Customize the viewer with additional options
            pdf_viewer(
                input=binary_data,
                width=700,
                height=1000,
                zoom_level=1.2,  # 120% zoom
                viewer_align="center",  # Center alignment
                show_page_separator=True  # Show separators between pages
            )
                
    elif st.session_state.last_uploaded_filename is not None: # File was removed by user
        # Also save session when file is removed
        save_session_to_json()
        st.session_state.extracted_introduction = None
        st.session_state.last_uploaded_filename = None
        st.session_state.pdf_file_bytes = None
        st.session_state.messages = [{"role": "assistant", "content": "Please upload a paper from the sidebar to start chatting."}]
        st.session_state.conversation_summary = None
        st.sidebar.info("PDF context removed.")
        # Rerun to update the chat display immediately
        st.rerun()

    st.sidebar.divider()
    st.sidebar.title("📥 Download Session")

    filename, session_data = _get_session_data_and_filename()

    if filename and session_data:
        try:
            json_string = json.dumps(session_data, indent=4, ensure_ascii=False)
            st.sidebar.download_button(
                label="Download Chat History",
                data=json_string,
                file_name=filename,
                mime="application/json",
                help="Download the current chat session as a JSON file."
            )
        except Exception as e:
            st.sidebar.error(f"Could not prepare download: {e}")
            st.sidebar.download_button(label="Download Chat History", data="", disabled=True)
    else:
        st.sidebar.download_button(
            label="Download Chat History",
            data="",
            file_name="chat_history.json",
            mime="application/json",
            help="Upload a paper and start a chat to download the session.",
            disabled=True
        )

    #st.sidebar.divider()
    #st.sidebar.title("📝 Summarize Conversation")

    # A summary can be generated if there is at least one user message
    user_messages_exist = any(m["role"] == "user" for m in st.session_state.get("messages", []))

    # if st.sidebar.button("Generate Lay Summary", disabled=not user_messages_exist, help="Generate a lay summary of the current conversation."):
    #     with st.spinner("Generating summary..."):
    #         try:
    #             # Prepare the conversation content for the summary prompt
    #             conversation_history = []
    #             for msg in st.session_state.get("messages", []):
    #                 # Exclude initial placeholder messages
    #                 if "upload a paper" not in msg["content"].lower():
    #                     conversation_history.append(f"{msg['role'].capitalize()}: {msg['content']}")
                
    #             conversation_text = "\n\n".join(conversation_history)

    #             # Prepare messages for the API call
    #             summary_prompt_content = f"{LAY_SUMMARY_PROMPT}\n\n{conversation_text}"
    #             api_messages_for_summary = [{"role": "user", "content": summary_prompt_content}]

    #             # Use the currently selected model for summarization
    #             config = MODEL_CONFIGS['GPT-3.5 Turbo']
    #             client = openai.OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    #             completion = client.chat.completions.create(
    #                 model=config["model"],
    #                 messages=api_messages_for_summary,
    #             )
    #             assistant_response = completion.choices[0].message.content
                
    #             st.session_state.conversation_summary = assistant_response
    #             st.rerun() # Rerun to display the summary text_area immediately
    #         except Exception as e:
    #             st.sidebar.error(f"Failed to generate summary: {e}")

    # # Display the summary if it exists
    # if "conversation_summary" in st.session_state and st.session_state.conversation_summary:
    #     st.sidebar.text_area("Conversation Summary", st.session_state.conversation_summary, height=250, key="summary_display")

st.write("LLM Science Explainers!")

loading_intro_from_pdf()

# Initialize session state to store received text
if 'text_received' not in st.session_state:
    st.session_state.text_received = []

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
                            "content": "[PAPER-TITLE]\n{}\n[PAPER]\n{}".format(st.session_state.last_uploaded_filename, st.session_state.extracted_introduction)
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
    # # Example of recording and playing back audio
    # st.write("Record your voice, and play the recorded audio:")
    # audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')
    # if audio:
    #     st.audio(audio['bytes'])

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
                
                # Prepare messages for OpenAI API, including the introduction and the default question.
                api_messages_for_openai = []

                api_messages_for_openai.append({"role": "assistant", "content": config["prompt"]})

                if st.session_state.extracted_introduction:
                    api_messages_for_openai.append({"role": "user", 
                                                    "content": "[PAPER-TITLE]\n{}\n[PAPER]\n{}".format(st.session_state.last_uploaded_filename, st.session_state.extracted_introduction)})

                # Get config for selected model
                print(f"Calling API for user prompt with model: {selected_model_key}")
                assistant_response = call_openai_api(api_messages_for_openai, selected_model_key)
                
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
