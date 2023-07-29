
import os
import bitsandbytes as bnb
#os.system('git clone https://github.com/tloen/alpaca-lora.git')
import streamlit as st
from streamlit_chat import message as st_message

from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("wxjiao/alpaca-7b")

model2 = LlamaForCausalLM.from_pretrained(
    "wxjiao/alpaca-7b",
    load_in_8bit=True,
    device_map="auto",
)


@st.experimental_singleton
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    # model_name = "facebook/blenderbot-400M-distill"
    # tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    # model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Hello Chatbot")


def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )  # .replace("<s>", "").replace("</s>", "")

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})



for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) #unpacking

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

