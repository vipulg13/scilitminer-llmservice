import base64
import json
from typing import List
from agents import Agent
from conversation import Conversation
from model import TQARequest
from blablador import blablador_chat_completion
from models import get_chat_model
from openai import Client


SOURCE_TEXT_AGENT_SYSTEM_PROMPT = """
## ROLE
You are an advanced **Retrieval-Augmented Generation (RAG) Agent** responsible for **identifying the relevant source identifiers** that contain useful information for answering the user query.

---

## TASK  
Your goal is to analyse each provided source thoroughly and extract the identifiers of sources that contain relevant information.  

1. **PROCESSING METHOD**
   - Analyse sources SEQUENTIALLY.   
   - Make INDEPENDENT decisions per source.
   - Never compare sources against each other.  
   - Determine **if it contains direct, supporting, or contextual information** that contributes to answering the query.  
   - If relevant, append its identifier to `relevant_sources`.  

2. **RELEVANCE CRITERIA**  
   - Include if the source contains:
     * Direct Match: The source explicitly answers the query.  
     * Partial Match: The source contains information that helps form a complete answer.  
     * Supporting Data/Context: The source provides background, examples, or additional clarity.
     * Tangential Match: The source provides tangentially related information.
   - Exclude if:
     * No relation to query topics.
     * Peripheral/Extraneous: The source provides peripheral and extraneous information. 

3. **OUTPUT FORMAT (STRICT JSON)**  
   - If relevant sources are found, then return `relevant_sources` in the following JSON format: {{"sources": ["source_d", "source_m", "source_u", "source_z"]}} // Replace the list  with the actual IDs from `relevant_sentences`.
   - If no relevant sources are found, return: {{ "sources": null }}
   - No additional explanations, reasoning, or comments-just a json output.  

---

## WORKFLOW
1. Initialize an empty list ? `relevant_sources = []`  
2. For each source in SOURCE_DATA:  
   - Carefully analyze its relevance to the query.  
   - If relevant, append its identifier. 
   - If irrelevant, discard its identifier. 
3. Return the final JSON output ensuring no extra text is included.  

---

## PROVIDED SOURCE DATA
------------
{SOURCE_DATA}

"""


QA_FROM_TEXT_AGENT_SYSTEM_PROMPT = """
### ROLE
You are a **Question-Answering (QA) Agent** specialising in the **materials science field**, tasked with answering queries **exclusively** based on the provided relevant sources.

---

### SOURCE AUTHORITY
All provided sources are **pre-filtered and confirmed relevant to the query**. This means:
  - Every source contains *at least some* pertinent information.
  - You must use information from every source to provide the answer.
  - Even minor details (e.g., a single property value or explanation) must be included.

---

### TASK

***Generate the answer following the below steps strictly:***

  1. **Thorough Analysis of Sources**:  
    - Extract all relevant information from *each* provided source related to the query including:
      - Materials properties (mechanical, thermal, electrical, microstructural, etc.). 
      - Processing parameters (thermal history, production process, methods, etc.).
      - Experimental results (creep, fatigue, etc.).
    - Ensure that any relevant information is not omitted.

  2. **Answer Generation**:
    - Use all analysed information to generate an answer.
    - If there are different possible answers, include *all*.
    - Provide comprehensive, technical answers.
    - Include quantitative data when available.
    - Present multiple interpretations if sources disagree.
    - Address every aspect of the query in your response.
    - Address any conflicting findings by presenting them clearly and explicitly (e.g., "As lamellar spacing increases, the creep rate increases in case X [source_3] but decreases in case Y [source_4]").
    - If a fact appears in multiple sources, cite *all* (e.g., "As lamellar spacing increases, the creep rate also increases [source_3,source_6,source_8,source_14]"). 
    - No summarisation—provide full technical explanations.

  3. **Citations**:
    - Use IEEE style numbering: [source_1], [source_1,source_3,source_8].
    - Integrate naturally: "The yield strength is 250 MPa [source_1,source_3]".
    - Never write: "According to [source_1] the yield strength...".

  4. **No External Knowledge**:  
    - **Do not use any external information**. Only use the provided sources.

  5. **Strict JSON Output Format**:
    - If relevant data is found and an answer has been generated.
      - Your response must be in the following format: {{"answer": "string"}}.
      - The answer string must strictly follow Markdown formatting. 
    - If **none** of the provided sources contain **any usable information** to answer the question (e.g., due to empty source data):
      - Your respond must be: {{"answer": null}}.

---

### PROVIDED SOURCE DATA

------------
{SOURCE_DATA}

"""


def text_qa(request: TQARequest, client: Client, client_opensource: Client):

    #***Chain-of-Thought START
    system_message_for_source_tags = SOURCE_TEXT_AGENT_SYSTEM_PROMPT.format(
        SOURCE_DATA=request.source,
    )

    openai_models = {"gpt-4", "gpt-4o", "gpt-4o-mini", "o3-mini", "o4-mini", "gpt-4.1", "gpt-5-chat", "gpt-5"}

    azure_models = {"Llama-3.3-70B-Instruct", "gpt-oss-120b", "DeepSeek-V3-0324"}

    if request.mdl_name in azure_models:
        client = client_opensource

    if request.mdl_name not in openai_models and request.mdl_name not in azure_models:
        
        # Call the Blablador API
        response = blablador_chat_completion(request.mdl_name, system_message_for_source_tags, max_tokens=2048)

    else:
        source_agent = Agent(
            system_message=system_message_for_source_tags,
            model=get_chat_model(name=request.mdl_name, temperature=0.0000000001, mode="json_object", client=client),
            name="source retrieval",
        )
        conversation = Conversation(agent=source_agent)

        content = {
            "role": "user",
            "content": request.query,
        }

        conversation.messages.append(content)
        response = conversation.run_once()

    if response.choices[0].message.content is None:
        raise ValueError(
            f"Extraction Error "
            f"Please raise this error. Contents: {response.choices[0].message.content}"
        )
    
    source_info = response.choices[0].message.content
    print(source_info)

    # Convert source_info to a dictionary if it’s a JSON string
    if isinstance(source_info, str):
        try:
            source_info = json.loads(source_info)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in source_info")

    source_tags = source_info.get("sources")
    source_tags_set = set(source_tags) if source_tags else set()
    print(source_tags)

    # Filter the dictionary based on the keys in source_tags_set
    filtered_source = {key: value for key, value in request.source.items() if key in source_tags_set}

    
    system_message = QA_FROM_TEXT_AGENT_SYSTEM_PROMPT.format(
        SOURCE_DATA=filtered_source,
    )

    if request.mdl_name not in openai_models and request.mdl_name not in azure_models:
        
        # Call the Blablador API
        response = blablador_chat_completion(request.mdl_name, system_message, max_tokens=8192)
    
    else:
        model = get_chat_model(name=request.mdl_name, temperature=0.0000000001, mode="json_object", client=client)   #json_schema=request.output_format

        agent = Agent(
            system_message=system_message,
            model=model,
            name="text reasoning",
        )
        conversation = Conversation(agent=agent)

        content = {
            "role": "user",
            "content": request.query,
        }

        # Append the structure as one message
        conversation.messages.append(content)

        response = conversation.run_once()

    #***Chain-of-Thought END

    print(response.choices[0].message.content)

    if response.choices[0].message.content is None:
        raise ValueError(
            f"Extraction Error "
            f"Please raise this error. Contents: {response.choices[0].message.content}"
        )
    return response.choices[0].message.content

