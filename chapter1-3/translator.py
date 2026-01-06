# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from tomlkit import document

from prompt import keyword_prompt

client = OpenAI(
    api_key='',
    base_url='https://api.deepseek.com'
)

def translate(sentence):
    '''
    Translate sentence(Chinese) into English
    :param sentence: a sentence to translate
    :return: translated sentence
    '''
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a translater"},
            {"role": "user", "content": sentence},
        ],
        stream=False
    )
    return response.choices[0].message.content

def keyword(document):
    '''
    Translate sentence(Chinese) into English
    :param sentence: a sentence to translate
    :return: translated sentence
    '''
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": keyword_prompt.format(document)},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    sentence = input("Please enter a sentence: ")
    print(keyword(sentence))
