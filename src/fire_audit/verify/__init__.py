"""
Verification subpackage for dataset manifests and splits.
"""

try:
    from src.fire_audit.verify.verifier import (
        CriterionResult,
        DatasetVerifier,
        VerificationReport,
        print_verification_report,
        verify_dataset_manifests,
    )
except ImportError:
    from fire_audit.verify.verifier import (
        CriterionResult,
        DatasetVerifier,
        VerificationReport,
        print_verification_report,
        verify_dataset_manifests,
    )

__all__ = [
    "CriterionResult",
    "DatasetVerifier",
    "VerificationReport",
    "print_verification_report",
    "verify_dataset_manifests",
]
