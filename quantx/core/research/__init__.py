"""Leakage-aware research data and validation infrastructure."""

from .causality import FeatureCausalityValidator, ReadWindow
from .data_version import DataVersion, DataVersionResolver, ProviderLock
from .dataset import DatasetBuilder, ResearchDataset
from .preprocessing import LabelWinsorizer
from .promotion import (
    PromotionCriteria,
    PromotionEvidence,
    PromotionGate,
    PromotionGateDecision,
)
from .schema import FeatureSchema
from .snapshot import PhysicalSnapshot, PhysicalSnapshotManager
from .specs import DatasetSpec, FeatureSpec, LabelSpec
from .split import AnchoredWalkForwardSplitter, ResearchFold
from .universe import PointInTimeUniverseProvider, UniverseAuditReport, UniverseAuditThresholds

__all__ = [
    "AnchoredWalkForwardSplitter",
    "DataVersion",
    "DataVersionResolver",
    "DatasetBuilder",
    "DatasetSpec",
    "FeatureCausalityValidator",
    "FeatureSpec",
    "FeatureSchema",
    "LabelSpec",
    "LabelWinsorizer",
    "PointInTimeUniverseProvider",
    "PhysicalSnapshot",
    "PhysicalSnapshotManager",
    "PromotionCriteria",
    "PromotionEvidence",
    "PromotionGate",
    "PromotionGateDecision",
    "ProviderLock",
    "ReadWindow",
    "ResearchDataset",
    "ResearchFold",
    "UniverseAuditReport",
    "UniverseAuditThresholds",
]
