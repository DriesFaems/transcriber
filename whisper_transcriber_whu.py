import streamlit as st
import tempfile
import os
import io
from pathlib import Path
import base64
import requests
import json
from openai import OpenAI
import subprocess
import shutil
from pydub import AudioSegment
import math

# Set page configuration
st.set_page_config(
    page_title="Audio/Video Transcription App",
    page_icon="🎙️",
    layout="wide"
)
# Check for API key
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = api_key

# Function to create a download link for text files
def get_download_link(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def split_audio(input_path, chunk_size_mb=24):
    """Split audio file into smaller chunks"""
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Calculate chunk size in milliseconds
        # We'll leave 1MB of headroom for the file format overhead
        file_size_bytes = os.path.getsize(input_path)
        duration_ms = len(audio)
        bytes_per_ms = file_size_bytes / duration_ms
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        chunk_duration_ms = int(chunk_size_bytes / bytes_per_ms)
        
        # Create chunks
        chunks = []
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            chunk.export(chunk_file.name, format="mp3")
            chunks.append(chunk_file.name)
        
        return chunks, None
    except Exception as e:
        return None, f"Error splitting audio: {str(e)}"

def transcribe_audio_with_api(file_path, model_name="whisper-1"):
    """Transcribe audio using OpenAI's Whisper API"""
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Open and read the file
    with open(file_path, "rb") as audio_file:
        # Call the OpenAI API for transcription
        response = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file
        )
    
    # Return the transcription text
    return response.text

def transcribe_large_file(file_path, model_name="whisper-1", progress_bar=None):
    """Handle large file transcription by splitting into chunks"""
    chunks, error = split_audio(file_path)
    if error:
        return None, error
        
    # Transcribe each chunk
    transcripts = []
    total_chunks = len(chunks)
    
    for i, chunk_path in enumerate(chunks):
        if progress_bar:
            progress_bar.progress((i / total_chunks) * 0.9, 
                                 text=f"Transcribing chunk {i+1}/{total_chunks}...")
            
        chunk_transcript = transcribe_audio_with_api(chunk_path, model_name)
        transcripts.append(chunk_transcript)
        os.unlink(chunk_path)  # Clean up
        
    if progress_bar:
        progress_bar.progress(0.95, text="Combining transcripts...")
        
    # Combine transcripts
    full_transcript = " ".join(transcripts)
    return full_transcript, None

# Main application
def main():
    st.title("Audio/Video Transcription App")
    
    # Sidebar for model selection
    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox(
        "Select Whisper Model",
        ["whisper-1"],
        index=0,
        help="OpenAI's Whisper API models"
    )
    
    # File uploader
    st.header("Upload your audio or video file")
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "mp4", "avi", "mov", "m4a", "webm"])
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB"
        }
        st.write(file_details)
        
        # Validate file size for API limits (25MB for Whisper API)
        if uploaded_file.size > 25 * 1024 * 1024:
            st.warning("File size exceeds the 25MB limit for OpenAI's Whisper API. The file will be split into chunks for processing.")
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process button
        if st.button("Transcribe"):
            try:
                # Check if file is over the API limit
                if uploaded_file.size > 25 * 1024 * 1024:
                    # Create a progress bar
                    progress_bar = st.progress(0, text="Processing large file...")
                    
                    # Split the audio file into chunks
                    with st.spinner("Splitting audio file into chunks..."):
                        transcript, error = transcribe_large_file(
                            tmp_file_path, 
                            model_name,
                            progress_bar=progress_bar
                        )
                        
                    if error:
                        st.error(f"An error occurred: {error}")
                        return
                    
                    progress_bar.progress(1.0, text="Complete!")
                else:
                    # File is within size limit, transcribe normally
                    with st.spinner(f"Transcribing with OpenAI's Whisper API..."):
                        transcript = transcribe_audio_with_api(tmp_file_path, model_name)
                
                # Display the transcript
                st.header("Transcript")
                st.text_area("Text", transcript, height=300)
                
                # Create download button for transcript
                st.markdown(
                    get_download_link(
                        transcript,
                        f"{Path(uploaded_file.name).stem}_transcript.txt",
                        "📥 Download Transcript as Text File"
                    ),
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()