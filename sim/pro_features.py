"""Public-core stubs for Orbital Engineering Lab Pro feature checks."""

FEATURE_OBJECT_PARALLELISM = "object_parallelism"
FEATURE_CAMPAIGNS = "campaigns"
FEATURE_CONSTELLATION_OPTIMIZATION = "constellation_optimization"
FEATURE_PRO_COMMUNICATIONS = "pro_communications"
FEATURE_CONTROLLER_BENCH = "controller_bench"
FEATURE_CUSTOMER_PACKAGES = "customer_packages"
FEATURE_GNC_WORKBENCH = "gnc_workbench"
FEATURE_SCALE_ANALYTICS = "scale_analytics"
FEATURE_TRAJECTORY_OPTIMIZATION = "trajectory_optimization"
FEATURE_VALIDATION_EVIDENCE = "validation_evidence"


def require_pro_feature(feature: str, **_kwargs):
    if feature in (FEATURE_CONTROLLER_BENCH, FEATURE_CUSTOMER_PACKAGES, FEATURE_GNC_WORKBENCH,
                   FEATURE_SCALE_ANALYTICS, FEATURE_TRAJECTORY_OPTIMIZATION, FEATURE_VALIDATION_EVIDENCE,
                   FEATURE_OBJECT_PARALLELISM, FEATURE_CAMPAIGNS, FEATURE_CONSTELLATION_OPTIMIZATION,
                   FEATURE_PRO_COMMUNICATIONS):
        return
    raise ImportError(
        f"Feature {feature!r} is available in Orbital Engineering Lab Pro and is not included in the public core."
    )
