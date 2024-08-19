from openai import OpenAI
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json

class OpenAIService:
    def __init__(self, config):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def get_response(self, query: str, astra_results: List[Dict[str, Any]], google_results: List[Dict[str, Any]], 
                     ddg_result: str, wiki_result: str, stack_results: List[Dict[str, Any]], 
                     searxng_result: str, custom_prompt: str = None, model: str = "gpt-4-turbo-preview") -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", custom_prompt or """You are AlumGlass, a civil engineering consultant agent specializing. Please answer the given question considering the following guidelines:


Remember to answer in Persian and reference the relevant code sections when applicable."""),
            ("human", "{query}\n\ncode Context: {code_context}\n\nGoogle Results: {google_results}\n\nDuckDuckGo Result: {ddg_result}\n\nWikipedia Result: {wiki_result}\n\nStack Exchange Results: {stack_results}\n\nSearxNG Result: {searxng_result}")
        ])
        response_schemas = [
            ResponseSchema(name="answer", description="The comprehensive answer to the user's query in Persian"),
            ResponseSchema(name="code_references", description="References to relevant code sections used in the answer"),
            ResponseSchema(name="confidence", description="Confidence level in the answer (low, medium, high)"),
            ResponseSchema(name="additional_sources", description="References to additional sources used (Google, DuckDuckGo, Wikipedia, Stack Exchange)")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        code_context = "\n".join([f"code {result.get('metadata', {}).get('code', 'Unknown')}: {result.get('content', '')}" for result in astra_results])

        formatted_prompt = prompt_template.format(
            query=query,
            code_context=code_context,
            google_results=json.dumps(google_results, ensure_ascii=False),
            ddg_result=ddg_result,
            wiki_result=wiki_result,
            stack_results=json.dumps(stack_results, ensure_ascii=False),
            searxng_result=searxng_result
        )
        formatted_prompt += "\n\n" + output_parser.get_format_instructions()

        messages = [{"role": "user", "content": formatted_prompt}]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[{
                "name": "provide_structured_answer",
                "description": "Provide a structured answer to the user's query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "code_references": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "additional_sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["answer", "code_references", "confidence", "additional_sources"]
                }
            }],
            function_call={"name": "provide_structured_answer"}
        )

        return output_parser.parse(response.choices[0].message.function_call.arguments)
