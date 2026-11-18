"""Backend-neutral output; token counts are never inferred from words."""
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int | None = None
    stats: dict = field(default_factory=dict)
