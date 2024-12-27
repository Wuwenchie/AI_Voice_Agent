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
struct  
pvrecorder  
wave  
pydub  
### Model  
Audio-to-text : whisper-large-v3  
Gemini AI : gemini-1.5-pro

## Step to run  
### 1. Setup Google api key  
### 2. Create a virtualenv  
    conda create --name AI python=3.11  
("AI" is the name of your virtualenv, you can use another name)  

    conda activate AI  
### 3. Install the library above using below command      
    pip install torch google.generativeai gtts transformers  
***
    pip install struct pvrecorder wave pydub  
### 4. Record your prompt  
If you just wnat to test the code, you can use the example .mp3 file and skip this step  

    python recorder.py  
### 5. Run the code using below command (choose one)    
    python voice_agent_en.py  
***
    python voice_agent_zh.py
