"""Bounded local OEL link-budget, coverage, scheduling, and validation workflows."""

from sim.pro_communications.availability import (
    StatisticalAggregateAvailabilityResult,
    StatisticalRainAvailabilityResult,
    evaluate_statistical_aggregate_availability,
    evaluate_statistical_rain_availability,
)
from sim.pro_communications.campaign import (
    LinkGeometryEvidence,
    TradeDimension,
    UncertaintyDimension,
    run_trade_campaign,
    run_uncertainty_campaign,
)
from sim.pro_communications.campaign_convergence import (
    CampaignConvergenceCriteria,
    CampaignConvergenceResult,
    evaluate_campaign_convergence,
)
from sim.pro_communications.contracts import (
    AdaptiveWaveformMode,
    AntennaPatternProfile,
    AtmosphericProfile,
    LinkBudgetProfile,
    ReceiverProfile,
    ResourceLimits,
    TransmitterProfile,
    WaveformProfile,
)
from sim.pro_communications.coverage import (
    ProCommunicationsCoverageConfig,
    ProCommunicationsCoverageResult,
    evaluate_pro_communications_coverage,
)
from sim.pro_communications.customer_adapter import (
    AdaptedCustomerProfiles,
    CustomerAdapterPolicy,
    adapt_customer_profile_file,
)
from sim.pro_communications.equipment_library import (
    GovernedEquipmentLibrary,
    load_governed_equipment_library,
)
from sim.pro_communications.link_budget import (
    ProLinkBudgetResult,
    ProLinkEnvironmentEvidence,
    evaluate_pro_link_budget,
)
from sim.pro_communications.network_trades import (
    GroundNetworkAlternative,
    GroundNetworkSite,
    ProDesignTradeResult,
    run_constellation_service_trade,
    run_ground_network_trade,
)
from sim.pro_communications.p837 import (
    P837_ANNEX_SOFTWARE_URL,
    P837_MODEL_EDITION,
    P837_REFERENCE_URL,
    P837MapSet,
    P837RainfallRateResult,
    apply_p837_rainfall_rate,
    evaluate_itu_p837_8_rainfall_rate,
    resolve_p837_map_set,
)
from sim.pro_communications.scenario_adapter import (
    CompletedScenarioObjectEvidence,
    ProScenarioFixedSiteLinkRequest,
    ProScenarioLinkProduct,
    evaluate_scenario_fixed_site_link,
    evaluate_scenario_rf_coverage,
    load_completed_scenario_object,
)
from sim.pro_communications.scheduling import (
    ProScheduleResult,
    generate_downlink_opportunities,
    schedule_downlink_opportunities,
)

__all__ = [
    "AdaptiveWaveformMode",
    "AdaptedCustomerProfiles",
    "AntennaPatternProfile",
    "AtmosphericProfile",
    "CustomerAdapterPolicy",
    "CampaignConvergenceCriteria",
    "CampaignConvergenceResult",
    "CompletedScenarioObjectEvidence",
    "GovernedEquipmentLibrary",
    "GroundNetworkAlternative",
    "GroundNetworkSite",
    "LinkGeometryEvidence",
    "LinkBudgetProfile",
    "ProCommunicationsCoverageConfig",
    "ProCommunicationsCoverageResult",
    "ProDesignTradeResult",
    "ProLinkBudgetResult",
    "ProLinkEnvironmentEvidence",
    "P837_ANNEX_SOFTWARE_URL",
    "P837_MODEL_EDITION",
    "P837_REFERENCE_URL",
    "P837MapSet",
    "P837RainfallRateResult",
    "ProScenarioFixedSiteLinkRequest",
    "ProScenarioLinkProduct",
    "ProScheduleResult",
    "ReceiverProfile",
    "ResourceLimits",
    "StatisticalRainAvailabilityResult",
    "StatisticalAggregateAvailabilityResult",
    "TradeDimension",
    "TransmitterProfile",
    "UncertaintyDimension",
    "WaveformProfile",
    "adapt_customer_profile_file",
    "apply_p837_rainfall_rate",
    "evaluate_pro_communications_coverage",
    "evaluate_campaign_convergence",
    "evaluate_pro_link_budget",
    "evaluate_itu_p837_8_rainfall_rate",
    "evaluate_scenario_fixed_site_link",
    "evaluate_scenario_rf_coverage",
    "evaluate_statistical_aggregate_availability",
    "evaluate_statistical_rain_availability",
    "generate_downlink_opportunities",
    "load_governed_equipment_library",
    "load_completed_scenario_object",
    "run_constellation_service_trade",
    "run_ground_network_trade",
    "run_trade_campaign",
    "run_uncertainty_campaign",
    "resolve_p837_map_set",
    "schedule_downlink_opportunities",
]
