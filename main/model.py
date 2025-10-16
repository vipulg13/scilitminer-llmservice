from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json


class TQARequest(BaseModel):
    mdl_name: str = "gpt-4o"
    query: str
    source: Dict[str, str]
    output_format: Dict = {
                            "name": "knowledge_reasoning",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                },
                                "required": ["answer"],
                                "additionalProperties": False,
                            }
                        }


class TQAResponse(dict):
    response: str
