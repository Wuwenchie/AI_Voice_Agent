# AI_Voice_Agent
**vedio of testing** --> https://youtu.be/fioo61fusmI   
### Requirements  
Python 3.11  
Google api key  -->  https://aistudio.google.com/app/apikey  
### Installation  
torch  
google.generativeai  
gtts  
transformers    
### Model  
Audio-to-text : whisper-large-v3  
Gemini AI : gemini-1.5-pro

## Step to run  
### Setup Google api key  
### Create a virtualenv and install the library above using below command  
    conda create --name AI python=3.11  
("AI" is the name of your virtualenv, you can use another name)  
    conda activate AI  
    
    pip install torch google.generativeai gtts transformers  
### Record your prompt  
python recorder.py  
### Run the code using below command  
python voice_agent_en.py (or python voice_agent_zh.py)
