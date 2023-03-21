import gradio as gr
import openai
#import pyttsx3
from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models and tokenizers
load_dotenv()
openai.api_key = "sk-"
gpt3_model = 'text-davinci-003'
gpt3_tokenizer = openai.Completion.create

dialogpt_model = 'microsoft/DialoGPT-medium'
dialogpt_tokenizer = AutoTokenizer.from_pretrained(dialogpt_model)
dialogpt_generator = AutoModelForCausalLM.from_pretrained(dialogpt_model)

# Define chatbot function
def get_model_reply(user_input, context=[]):
  # Choose a model randomly (you can use other criteria here)
  model_choice = random.choice(["gpt3_model", "dialogpt_model"])

  # Generate response using chosen model
  if model_choice == "gpt3_model":
    # Use gpt-3 model
    context += [user_input]
    completion = openai.Completion.create(
      model='text-davinci-003',
      prompt="\n".join([f"I am {role}.", *context])[:4096],
      max_tokens=1048,
      temperature=0.9,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
    )
    response = completion.choices[0].text.strip("\n")
    context += [response]

  else:
    # Use dialogpt model
    new_user_input_ids = dialogpt_tokenizer.encode(user_input + dialogpt_tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([context[-1], new_user_input_ids], dim=-1) if len(context) > 0 else new_user_input_ids  
    chat_history_ids = dialogpt_generator.generate(bot_input_ids, max_length=1000,pad_token_id=dialogpt_tokenizer.eos_token_id)
    response = dialogpt_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    context.append(chat_history_ids)

  # Convert text to speech using pyttsx3
 # engine = pyttsx3.init()
 # engine.say(response)
 # engine.runAndWait()

  # Return response and context as output
  return response, context

# Create web interface using gradio
#sate = gr.State()
#ui = gr.Interface(fn=chatbot ,inputs=[gr.Audio(source="microphone",type="filepath"), state], outputs=[gr.Textbox(), state])
#ui.launch()

#Second interface
with gr.Blocks() as dialog_app:
    chatbot = gr.Chatbot() # dedicated "chatbot" component
    state = gr.State([]) # session state that persists across multiple submits
    
    with gr.Row():
        txt = gr.Textbox(
            show_label=False, 
            placeholder="Enter text and press enter"
        ).style(container=False)

    txt.submit(get_model_reply, [txt, state], [chatbot, state])

# launches the app in a new local port
dialog_app.launch()
