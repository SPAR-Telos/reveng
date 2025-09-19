"""Pydantic models for LLM judge scoring responses."""

from typing import List

from pydantic import BaseModel, Field


class CategoryScore(BaseModel):
    """Score for a specific category in individual trajectory scoring."""

    category: str = Field(description="Name of the scoring category")
    points: int = Field(ge=0, le=100, description="Points awarded for this category")
    max_points: int = Field(
        ge=0, le=100, description="Maximum possible points for this category"
    )
    explanation: str = Field(description="Brief explanation of the score")


class IndividualScoringResponse(BaseModel):
    """Response model for individual trajectory scoring."""

    overall_score: int = Field(ge=0, le=100, description="Total score out of 100")
    category_scores: List[CategoryScore] = Field(description="Scores for each category")
    detailed_reasoning: str = Field(
        description="Comprehensive analysis of the trajectory"
    )
    key_strengths: List[str] = Field(description="List of 2-3 main strengths")
    areas_for_improvement: List[str] = Field(
        description="List of 2-3 main areas for improvement"
    )
    final_assessment: str = Field(
        description="Summary assessment of trajectory quality"
    )


class TrajectoryComparison(BaseModel):
    """Comparison details for a single trajectory."""

    trajectory_name: str = Field(description="Name of the trajectory")
    rank: int = Field(ge=1, description="Ranking position (1 is best)")
    justification: str = Field(description="Brief justification for this ranking")


class CategoryAnalysis(BaseModel):
    """Analysis for a specific comparison category."""

    category: str = Field(description="Name of the comparison category")
    analysis: str = Field(
        description="Detailed analysis comparing trajectories in this category"
    )


class TrajectoryComparisonResponse(BaseModel):
    """Response model for trajectory comparison."""

    ranking: List[TrajectoryComparison] = Field(
        description="Ranked list of trajectories"
    )
    category_analyses: List[CategoryAnalysis] = Field(
        description="Analysis for each comparison category"
    )
    key_differentiators: List[str] = Field(
        description="Main factors that distinguish the trajectories"
    )
    winner_justification: str = Field(
        description="Comprehensive explanation for the top-ranked trajectory"
    )
    recommendations: List[str] = Field(
        description="Improvement suggestions for each trajectory"
    )
    overall_assessment: str = Field(description="Summary of the comparative analysis")
