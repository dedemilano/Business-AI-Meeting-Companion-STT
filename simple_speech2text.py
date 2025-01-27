import requests 

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3"

response = requests.get(url)

audio_file_path = "downloaded_audio.mp3"

if response.status_code == 200:
    with open(audio_file_path , "wb") as file:
        file.write(response.content)
    print("Audio file downloaded")
else:
    print("Error downloading audio file")

import torch 
from transformers import pipeline 


pipe = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-tiny.en",
    chunk_length_s = 30,
)

sample = audio_file_path

prediction = pipe(sample , batch_size=8)["text"]

print(prediction)