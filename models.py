from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class Violation:
    """A single violation extracted from an audit act."""

    raw_text: str
    source_document: str
    page: int
    section: str
    description: str
    subject: str
    law_ref: str

    id: str = field(default="")
    normalized_text: str = field(default="")
    keywords: List[str] = field(default_factory=list)
    evidence_score: Optional[float] = field(default=None)
    legal_score: Optional[float] = field(default=None)
    actionability_score: Optional[float] = field(default=None)
    evidence_comment: str = field(default="")
    legal_comment: str = field(default="")
    actionability_comment: str = field(default="")
    possibly_not_a_violation: bool = field(default=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class RetrievalResult:
    """A single result from a vector/BM25 search."""

    chunk_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViolationContext:
    """All retrieval results for a single violation — immutable after creation."""

    violation: Violation
    ref_results: Dict[str, List[RetrievalResult]] = field(default_factory=dict)
    retrieval_queries: Dict[str, str] = field(default_factory=dict)
    used_chunk_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.used_chunk_ids and self.ref_results:
            self.used_chunk_ids = [
                r.chunk_id
                for results in self.ref_results.values()
                for r in results
            ]


@dataclass
class ComparisonResult:
    """Similarity match between a violation and a reference item."""

    match_id: str
    source_type: str
    similarity_score: float
    matched_text: str
    law_ref: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovedFormulation:
    """LLM-generated improvement for a violation formulation."""

    improved_text: str
    legal_qualification: str
    justification: str
    recommendation: str
    legal_qualification_grounded: bool = field(default=False)
    norm_source_ids: List[str] = field(default_factory=list)
    agent_output_raw: str = field(default="")


@dataclass
class ItemTrace:
    """Full traceability record for a checklist item."""

    violation_id: str
    used_chunk_ids: List[str] = field(default_factory=list)
    retrieval_queries: Dict[str, str] = field(default_factory=dict)
    evidence_sources: List[str] = field(default_factory=list)
    norm_sources: List[str] = field(default_factory=list)
    typical_sources: List[str] = field(default_factory=list)
    verifier_notes: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of a deterministic verifier check."""

    axis: str
    status: str
    detail: str
    pattern_count: int = field(default=0)
    matched_patterns: List[str] = field(default_factory=list)


@dataclass
class ChecklistItem:
    """Final assembled checklist item for a single violation."""

    violation_id: str
    raw_text: str
    description: str
    subject: str
    law_ref: str
    source_document: str
    page: int
    section: str

    evidence_score: Optional[float] = field(default=None)
    legal_score: Optional[float] = field(default=None)
    actionability_score: Optional[float] = field(default=None)
    confidence_score: float = field(default=0.0)

    evidence_status: str = field(default="unknown")
    legal_status: str = field(default="unknown")
    actionability_status: str = field(default="unknown")

    evidence_comment: str = field(default="")
    legal_comment: str = field(default="")
    actionability_comment: str = field(default="")

    improved_formulation: str = field(default="")
    legal_qualification: str = field(default="")
    legal_qualification_grounded: bool = field(default=False)
    justification: str = field(default="")
    recommendation: str = field(default="")

    comparisons: Dict[str, List[ComparisonResult]] = field(default_factory=dict)
    verification_notes: List[VerificationResult] = field(default_factory=list)
    agent_outputs: Dict[str, str] = field(default_factory=dict)

    possibly_not_a_violation: bool = field(default=False)
    trace: Optional[ItemTrace] = field(default=None)
