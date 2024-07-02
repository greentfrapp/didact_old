from pydantic import (
    BaseModel,
    Field,
)


class Embeddable(BaseModel):
    embedding: list[float] | None = Field(default=None, repr=False)


class EmbeddedText(Embeddable):
    text: str
    source: str | None = None
