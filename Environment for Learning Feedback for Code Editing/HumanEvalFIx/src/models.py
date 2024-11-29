import openai
from abc import ABC, abstractmethod
import asyncio
from functools import partial
from tqdm.asyncio import tqdm_asyncio
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class AbstractModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        
    @abstractmethod
    def _set_model(self, args):
        pass


    @abstractmethod
    def async_generate(self, model_input, **kwargs):
        pass

    @abstractmethod
    def generate_concurrently(self, all_model_data):
        pass

class OpenAIModel(AbstractModel):
    def __init__(self, model_name="gpt-3.5-turbo-1106", mode="editor", temperature=0.1, top_p=0.9, frequency_penalty=0.0):
        """
        Set the openai key and organization on the script.
        """
        super().__init__(model_name)
        self.mode = mode
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.model = self._set_model()
        self.total_tokens = 0
    
    def _set_model(self):
        return ChatOpenAI(
            model_name=self.model_name,  # 'gpt-3.5-turbo' or 'gpt-4'
            temperature=self.temperature,
            max_tokens=1024,
            max_retries=200,
            request_timeout=60,
            model_kwargs = {
                'stop':['\n###', '\n```python', 'Here is the corrected code:', 'The corrected code is:'] if self.mode == "critic" else [],
                'top_p': self.top_p,
                'frequency_penalty': self.frequency_penalty,
            }
        )
    
    async def async_generate(self, model_input):
        while True:
            try:
                system_message = HumanMessage(content= model_input)
                response = await self.model.agenerate([[system_message]])
                break
            
            except Exception as e:
                print(f"Exception occurred: {e}")
                response = None
                break

        return response.generations[0][0].text
    
    async def generate_concurrently(self, all_model_data):
        tasks = [self.async_generate(model_data)
                for i, model_data in enumerate(all_model_data)]
        return await tqdm_asyncio.gather(*tasks)
    
    
class VllmModel(AbstractModel):
    def __init__(self, model_name="critic", mode="editor", port=8000, url="localhost", temperature=0.1, top_p=0.9, frequency_penalty=0.0):
        """
        Set the openai key and organization on the script.
        """
        super().__init__(model_name)
        self.mode = mode
        self.url = url
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.model = self._set_model(port)
    
    def _set_model(self, port):
        return OpenAI(
            model_name=self.model_name,
            openai_api_base = f"http://{self.url}:{port}/v1",
            openai_api_key="EMPTY",
            max_retries=3,
            max_tokens=1024,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            model_kwargs={
                    'stop': ['\n###', '\n```', 'Here is the corrected code:', 'The corrected code is:'] if self.mode == "critic" else ['##']
                }
        )
    
    async def async_generate(self, model_input):
        while True:
            try:
                response = await self.model.agenerate([model_input])
                # print(response)
                break
            
            except Exception as e:
                print(f"Exception occurred: {e}")
                response = None
                break

        ## TODO : change this code if you want to save it in a different way ##
        return response.generations[0][0].text

    async def generate_concurrently(self, all_model_data):
        tasks = [self.async_generate(model_data)
                for i, model_data in enumerate(all_model_data)]
        return await tqdm_asyncio.gather(*tasks)