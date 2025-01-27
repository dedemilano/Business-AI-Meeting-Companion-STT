import torch 
from transformers import pipeline
import gradio as gr


def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model = "openai/whisper-tiny.en",
        chunk_length_s = 30,
    )

    result = pipe(audio_file , batch_size=8)["text"]

    return result 

audio_input = gr.Audio(sources="upload" , type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio , 
                     inputs = audio_input , 
                     outputs = output_text , 
                     title="Audio Transcription App" , 
                     description="Upload the audio file")

iface.launch(server_name="127.0.0.1" , server_port=7860)