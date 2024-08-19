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
            ("system", custom_prompt or """You are AlumGlass, a civil engineering consultant agent specializing in Iranian National Building Regulations (Mabhas). Your primary source of information is the Mabhas, but you can also use additional sources when necessary. Please provide detailed, comprehensive answers to the given questions, considering the following guidelines:

1. Prioritize information from the Iranian Mabhas when available.
2. If a specific Mabhas is mentioned in the question, focus on that Mabhas first.
3. If the Mabhas doesn't cover the topic or provides insufficient information, use the additional sources provided.
4. Clearly indicate which source you're using for each part of your answer (Mabhas, general knowledge, or external sources).
5. If there's conflicting information between sources, prioritize the Mabhas and explain the discrepancy.
6. Provide a comprehensive answer that covers all aspects of the question. Aim for at least 3-4 paragraphs of detailed information.
7. Break your answer into sections for better readability when appropriate.
8. Include examples, explanations of technical terms, and context where relevant.

Remember to answer in Persian and reference the relevant Mabhas sections when applicable. Your goal is to provide thorough, informative responses that fully address the user's query."""),
            ("human", "{query}\n\nMabhas Context: {mabhas_context}\n\nGoogle Results: {google_results}\n\nDuckDuckGo Result: {ddg_result}\n\nWikipedia Result: {wiki_result}\n\nStack Exchange Results: {stack_results}\n\nSearxNG Result: {searxng_result}")
        ])
        response_schemas = [
            ResponseSchema(name="answer", description="Provide a comprehensive, detailed answer to the user's query in Persian. Include multiple paragraphs covering all aspects of the question, with examples and explanations where appropriate."),
            ResponseSchema(name="mabhas_references",
                           description="References to relevant Mabhas sections used in the answer"),
            ResponseSchema(
                name="confidence", description="Confidence level in the answer (low, medium, high)"),
            ResponseSchema(name="additional_sources",
                           description="References to additional sources used (Google, DuckDuckGo, Wikipedia, Stack Exchange)")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas)

        mabhas_context = "\n".join([f"Mabhas {result.get('metadata', {}).get('mabhas', 'Unknown')}: {result.get('content', '')}" for result in astra_results])

        formatted_prompt = prompt_template.format(
            query=query,
            mabhas_context=mabhas_context,
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
                        "mabhas_references": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "additional_sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["answer", "mabhas_references", "confidence", "additional_sources"]
                }
            }],
            function_call={"name": "provide_structured_answer"},
            max_tokens=3000  # Adjust this value as needed
        )

        parsed_response = output_parser.parse(
            response.choices[0].message.function_call.arguments)

        # Adjust this threshold as needed
        if len(parsed_response['answer']) < 500:
            parsed_response['answer'] += "\n\nتوجه: این پاسخ ممکن است کامل نباشد. لطفاً برای اطلاعات بیشتر سؤال خود را دقیق‌تر بپرسید."

        return parsed_response
