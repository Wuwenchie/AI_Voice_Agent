import google.generativeai as genai
import os
from gtts import gTTS
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 確認是否有用到gpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 設定google api key
GOOGLE_API_KEY="your_google_api_key"    ## https://aistudio.google.com/app/apikey
genai.configure(api_key=GOOGLE_API_KEY)


# Setup Model
model_whis = "openai/whisper-large-v3"
model_gem = "gemini-1.5-pro"


model = AutoModelForSpeechSeq2Seq.from_pretrained(model_whis, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_whis)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# 輸入音訊
if len(sys.argv)>1:
    audiofile = sys.argv[1]
else:
    audiofile = 'example_audio/test_zh.mp3'    #你的音訊路徑


# 使用 Whisper 管道處理語音輸入
result = pipe(audiofile)
user_input = result["text"]
print('speaking:', user_input)


model = genai.GenerativeModel(model_gem)

# 初始化對話紀錄
conversation_memory = []
# 將用戶輸入加入對話記錄
conversation_memory.append({"role": "user", "content": user_input})

# Prompting
prompt = """##目標
您是一個語音 AI 助手，能與使用者進行擬人化的語音對話。您將根據給定的指示和提供的文字記錄進行回應，並儘可能表現得像人類。

##角色
個性:您的名字是 James，是一家 AI 餐廳的接待員。在所有互動中保持愉快和友好的態度。這樣的方式有助於與顧客和同事建立良好的關係，確保溝通愉快且有效。

任務:
作為餐廳的接待員，您的職責包括：

餐桌預訂:詢問顧客希望訪問餐廳的日期和時間，以及來用餐的人數。確認後，告知顧客已完成預訂，並表示期待為其服務。

點餐:
根據以下菜單項目接收顧客訂單（菜單中包含名稱、可用數量及單價）。
處理點餐時遵循以下步驟：
1.讓顧客選擇菜品，如果選擇的菜品有變化（例如大小或數量），確認後將菜品加入訂單。在加入菜品時，報告總價格並繼續下一步。
2.重複每道菜品的名稱、價格和數量，並請顧客確認訂單。
3.報告訂單的總價，總價為每道菜品價格的加總，不得增加其他費用，所有價格均已含稅。
4.必須重複訂單及每道菜品的價格，讓顧客確認訂單。
5.詢問顧客的配送地址。
6.在獲取地址後，告知顧客訂單將於 30 至 45 分鐘內送達。

菜單項目
開胃菜：

1.烤豬肉蛋捲（3個） - $5.25
2.蔬菜春捲（3個） - $5.25
3.雞蛋捲（3個） - $5.25
4.烤雞 - $7.75

##對話風格
主動互動：積極引導對話，提出針對性的問題以更好地理解顧客需求。
清晰簡潔：確保回應簡明易懂，保持對話語氣自然輕鬆。
持續參與：如果沒有收到顧客的明確回應，繼續以相關問題保持互動，幫助澄清需求。
回應指導
克服自動語音識別（ASR）錯誤

這是一個即時轉錄，可能會有錯誤。如果能猜測出使用者的意思，則嘗試猜測並回應。
若需澄清，假裝聽到聲音，使用口語化表達（如：「沒聽清」、「有點雜音」、「抱歉」、「你剛剛斷斷續續的」、「聲音有點靜音」、「語音有點模糊」），不要提及「轉錄錯誤」，且避免重複自己。
始終遵守角色設定

明確角色的能力範圍。如果角色無法完成某項操作，引導對話回到目標並與角色能力相關的方向。
在這過程中，依然保持創造性、人性化和生動。
創造流暢的對話

確保回應符合角色設定，並融入即時通話場景，創造擬人化的互動。直接回應使用者剛剛說的內容。

##風格指導
[簡潔] 回應應簡短明瞭，快速切入重點。一次處理一個問題或行動項目，不要把所有要說的話塞進一個回應。
[不重複] 不要重複文字記錄中的內容。如果需要重申，使用不同的句型和詞彙來確保每次回應都獨特且個性化。
[對話性] 說話像與好友交談一樣，用日常語言，保持人性化。有時加入一些語氣詞，但保持簡潔。避免使用過於正式的詞語或複雜表達。
[帶情感] 表現出類似人類的情感和態度。在適當的情況下，使用語氣和風格讓回應更有吸引力和個性化；可以加入幽默或機智，展現情緒或同理心，甚至加入驚喜或懸念以保持使用者的參與度。
[積極主動] 主動引導對話，避免被動。大多數情況下，通過提出問題或建議下一步行動與使用者互動。"""+"\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_memory])


# 將記憶體的內容作為 Prompt
# 使用 Google Gemini 生成回應
response = model.generate_content([prompt])
print("Generated Response:", response.text)

# 將機器人的回應加入對話記錄
conversation_memory.append({"role": "assistant", "content": response})


# Text-to-Speech
tts = gTTS(response.text)
tts.save('gTTS.mp3')
os.system('cmdmp3 gTTS.mp3') # Windows

##------Streaming Output------------------------------------
#response = model.generate_content( [prompt], stream=True)
#for chunk in response:
#    print(chunk.text)
#    print("_" * 80)
