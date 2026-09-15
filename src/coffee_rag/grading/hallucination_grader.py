"""
Groundedness / hallucination grading, cheap-first:

  primary    MiniCheckGrader  -- local Flan-T5, P(answer supported by context)
  secondary  LLMJudgeGrader   -- gpt-4.1-mini, binary supported/not + reason

HallucinationGrader.grade() always runs the primary; when its score is
ambiguous and the tie-breaker is enabled, it defers to the secondary.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from coffee_rag.config import settings
from coffee_rag.templates import HL_GRADER_PROMPT


@dataclass
class GroundednessResult:
    score: float                # MiniCheck P(supported)
    low_confidence: bool        # -> schemas.AnswerResult.low_confidence
    judged_by: str              # "minicheck" or "llm"
    reason: str | None = None   # the judge's one-liner, when it ran


# --- primary: MiniCheck ---------------------------------------------------

class MiniCheckGrader:
    def __init__(self, model=None) -> None:
        if model is None:
            from minicheck.minicheck import MiniCheck

            model = MiniCheck(model_name=settings.HALLUCINATION_MODEL)
        self.model = model

    def score(self, answer: str, context: str) -> float:
        _, probs, _, _ = self.model.score(docs=[context], claims=[answer])
        return float(probs[0])


# --- secondary: LLM as a judge ------------------------------------------

class _Grade(BaseModel):
    supported: bool = Field(
        description="True only if every factual claim in the answer is verifiable "
        "from the context alone."
    )
    reason: str = Field(
        description="One sentence: the unsupported claim, or why it all checks out."
    )

class LLMJudgeGrader:
    def __init__(self, llm=None) -> None:
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=settings.HALLUCINATION_TIEBREAKER_MODEL,
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
            )
        
        # HL_GRADER_PROMPT is a template from ChatPromptTemplate.from_messages    
        self.system_prompt = HL_GRADER_PROMPT
        self.chain = llm.with_structured_output(_Grade)

    def judge(self, answer: str, context: str) -> _Grade:
        return self.chain.invoke(
            HL_GRADER_PROMPT
        )


# --- coordinator --------------------------------------------------------
class HallucinationGrader:
    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = primary or MiniCheckGrader()
        self.secondary = secondary or LLMJudgeGrader()

    def grade(self, answer: str, context: str) -> GroundednessResult:
        score = self.primary.score(answer, context)
        lo, hi = settings.HALLUCINATION_AMBIGUOUS_LOW, settings.HALLUCINATION_AMBIGUOUS_HIGH

        if lo <= score <= hi and settings.HALLUCINATION_TIEBREAKER_ENABLED:
            g = self.secondary.judge(answer, context)
            return GroundednessResult(
                score, low_confidence=not g.supported, judged_by="llm", reason=g.reason
            )

        return GroundednessResult(
            score,
            low_confidence=score < settings.HALLUCINATION_SCORE_THRESHOLD,
            judged_by="minicheck",
        )
  