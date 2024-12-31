# AI_Voice_Agent
### introduction  
Managing calls in a restaurant, hospital, or any service-oriented business can be a daunting task. The unpredictability of call volumes often leaves businesses with two options both of which are not ideal: overstaffing or making customers wait in long queues. Overstaffing drives up operational costs, while long wait times frustrate customers, reducing overall satisfaction.  
This AI voice agent can be utilized for making orders in the restaurant. With the .mp3 file of recorder, it will generate a precisely response quickly.  
### vedio of testing   
voice_agent_zh.py --> https://youtu.be/fioo61fusmI   
voice_agent_gr.py --> https://youtube.com/shorts/eVWUdsj1nvU   
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
gradio  
numpy  
### Model  
Audio-to-text : whisper-large-v3  
Gemini AI : gemini-1.5-pro

## Step to run  
### voice_agent_en.py OR voice_agent_zh.py
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
***
### voice_agent_gr.py
### 1. Run the code
    gradio voice_agent_gr.py  
### 2. Open the URL 
for example, http://127.0.0.1:7860/    
### 3. Try it !!
