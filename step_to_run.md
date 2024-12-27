**Setup Google api key**  
**Create a virtualenv and install the library above using below command**  
conda create --name AI python=3.11 ("AI" is the name of your virtualenv, you can use another name)  
conda activate AI  
pip install torch google.generativeai gtts transformers  
**Record your prompt**  
python recorder.py  
**Run the code using below command**  
python voice_agent_en.py (or python voice_agent_zh.py)
