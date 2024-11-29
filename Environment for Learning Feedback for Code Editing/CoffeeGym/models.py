import openai
from abc import ABC, abstractmethod
import asyncio
from functools import partial
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from utils.path import load_prompt

class AbstractModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        
    @abstractmethod
    def _set_model(self):
        pass

    @abstractmethod
    def get_prompt_template(self, dataset_name):
        pass

    @abstractmethod
    def apply_function_to_prompt(self, prompt_template, mode, **kwargs):
        pass

    @abstractmethod
    def inference(self, prompt):
        pass

class OpenAIModel(AbstractModel):
    def __init__(self, model_name="gpt-3.5-turbo-1106"):
        """
        Set the openai key and organization on the script.
        """
        super().__init__(model_name)
        self.model = self._set_model()
        self.total_tokens = 0
    
    def _set_model(self):
        return ChatOpenAI(
            model_name=self.model_name,
            max_retries=3,
        )

    def get_total_cost(self):
        cost_per_tokens = {
            "gpt-3.5-turbo-1106": 0.00000035,
            "gpt-4": 0.00000035,
        }
        return self.total_tokens * cost_per_tokens[self.model_name]
    
    
    def get_prompt_template(self, mode):
        """Get a prompt template based on the model name."""
        prompt_info = load_prompt(self.model_name)[mode]
        return PromptTemplate(input_variables=prompt_info["input_variables"], template=prompt_info["prompt"])

    def apply_function_to_prompt(self, prompt_template, mode, **kwargs):
        """Apply function to the prompt template."""
        
        if mode == "refine":
            prompt = prompt_template.format(description=kwargs["description"], output_format=kwargs["output_format"], input_format=kwargs['input_format'], wrong_code=kwargs["wrong_code"], feedback=kwargs["feedback"])
        elif mode == "feedback":
            prompt = prompt_template.format(description=kwargs["description"], output_format=kwargs["output_format"], input_format=kwargs['input_format'], wrong_code=kwargs["wrong_code"])
        else: # initial   
            prompt = prompt_template.format(**kwargs)
        return prompt

    def inference(self, prompt, **kwargs):
        """
        Perform inference using the specified model.
        **kwargs are additional generation arguments to be passed to the model.
        example) 
            temperature, max_tokens, top_p, frequency_penalty, stop, n
        """
        
        response = []
        for i in range(kwargs.get("n", 1)):
            with get_openai_callback() as callback:
                org_response = self.model.invoke(
                    prompt,
                    **kwargs
                )
            self.total_tokens += callback.total_tokens  
            response.append(org_response.content)
            
        return response

    async def async_inference(self, prompt, **kwargs):
        """
        Perform asynchronous inference using the specified model.
        **kwargs are additional generation arguments to be passed to the model.
        example) 
            temperature, max_tokens, top_p, frequency_penalty, stop, n
        """
        n = kwargs.get("n", 1)
        loop = asyncio.get_event_loop()

        async def fetch():
            func = partial(self.model.invoke, prompt, **kwargs)
            response = await loop.run_in_executor(None, func)
            return response.content

        tasks = [fetch() for _ in range(n)]
        responses = await asyncio.gather(*tasks)

        return responses
    
    
class VllmModel(AbstractModel):
    def __init__(self, model_name="critic", port=8000, url="http://localhost"):
        """
        Set the openai key and organization on the script.
        """
        super().__init__(model_name)
        self.url = url
        self.org_model_name = self._set_model_org_name()
        self.model = self._set_model(port)
        self.dataset_name = None
        print(f"model name: {self.model_name}, org_model_name: {self.org_model_name}")
    
    def _set_model(self, port):
        return OpenAI(
            model_name=self.model_name,
            openai_api_base = f"{self.url}:{port}/v1" if self.org_model_name != "COFFEEPOTS-editor" else self.url,
            openai_api_key="EMPTY",
            max_retries=3
        )
    
    def _set_model_org_name(self):
        if "OpenCode" in self.model_name:
            return "OpenCodeInterpreter"
        elif "COFFEEPOTS-critic" in self.model_name:
            return "COFFEEPOTS-critic"
        elif "COFFEEPOTS-editor" in self.model_name:
            return "COFFEEPOTS-editor"
        elif "ours_and_oasst" in self.model_name:
            return "ours_w_oasst"
        elif "only_ours" in self.model_name:
            return "only_ours"
        elif "deepseek" in self.model_name or "editor" in self.model_name:
            return "deepseek_reward"
        else:
            return "deepseek_reward"
    
    def get_model_name(self):
        return self.org_model_name
    
    def get_prompt_template(self, mode):
        """Get a prompt template based on the dataset name."""
        prompt_info = load_prompt(self.org_model_name)[mode]
        return PromptTemplate(input_variables=prompt_info["input_variables"], template=prompt_info["prompt"])
    
    
    def apply_function_to_prompt(self, prompt_template, mode, **kwargs):
        """Apply function to the prompt template."""
        
        if mode == "refine":
            prompt = prompt_template.format(description=kwargs["description"], output_format=kwargs["output_format"], input_format=kwargs['input_format'], wrong_code=kwargs["wrong_code"], feedback=kwargs["feedback"])
        elif mode == "feedback":
            prompt = prompt_template.format(description=kwargs["description"], output_format=kwargs["output_format"], input_format=kwargs['input_format'], wrong_code=kwargs["wrong_code"])
        else: # initial   
            prompt = prompt_template.format(**kwargs)
        return prompt

    def inference(self, prompt, **kwargs):
        """
        Perform inference using the specified model.
        **kwargs are additional generation arguments to be passed to the model.
        example) 
            temperature, max_tokens, top_p, frequency_penalty, stop, n
        """
        
        response = self.model.invoke(
            prompt,
            **kwargs
        )
        
        # Handle multiple responses if n > 1
        if isinstance(response, list):
            return response
        else:
            return [response]
    
    async def async_inference(self, prompt, **kwargs):
        """
        Perform asynchronous inference using the specified model.
        **kwargs are additional generation arguments to be passed to the model.
        example) 
            temperature, max_tokens, top_p, frequency_penalty, stop, n
        """
        # loop = asyncio.get_event_loop()
        # print(kwargs)
        # func = partial(self.model.invoke, prompt, **kwargs)
        # response = await loop.run_in_executor(None, func)
        response = await self.model.agenerate(prompts=[prompt], **kwargs)
        response = [g.text for g in response.generations[0]]
        # print(response)
        
        # Handle multiple responses if n > 1
        if isinstance(response, list):
            return response
        else:
            return [response]

    
