import os
import gradio as gr
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import InferenceClient

load_dotenv()  # Load environment variables from .env file

# Initialize the Hugging Face client
client = InferenceClient(
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def analyze_text(text):
    """Analyze text using a smaller, suitable model"""
    try:
        response = client.text_generation(
            text,
            model="mistralai/Mistral-7B-Instruct-v0.2",  # Changed to a smaller, instruction-tuned model
            max_new_tokens=512,
            temperature=0.7,
            return_full_text=False
        )
        return response
    except Exception as e:
        print(f"Text generation error: {str(e)}")
        return f"Error in text analysis: {str(e)}"

def transcript_audio(audio_file):
    """Process audio file and analyze its transcript"""
    if audio_file is None:
        return "Please upload an audio file."
        
    try:
        # Initialize speech recognition pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            chunk_length_s=30
        )
        
        # Transcribe audio
        transcript = pipe(audio_file, batch_size=8)["text"]
        
        if not transcript:
            return "No speech detected in the audio file."
        
        # Create prompt for analysis
        prompt = f"""<s>[INST] Please analyze this transcript and provide key points with details:

Transcript:
{transcript}

List the main points, important details, and any notable insights from this transcript. [/INST]"""
        
        # Analyze transcript
        result = analyze_text(prompt)
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error details: {error_msg}")  # For debugging
        return f"An error occurred while processing the audio: {error_msg}"

# Create Gradio interface
interface = gr.Interface(
    fn=transcript_audio,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs=gr.Textbox(),
    title="Audio Transcription and Analysis",
    description="Upload an audio file to transcribe and analyze its content"
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)