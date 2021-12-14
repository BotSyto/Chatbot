from pydantic import BaseModel


class NLPProto(BaseModel):
    query: str