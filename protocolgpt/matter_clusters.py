"""
Matter Cluster Definitions
Defines all Matter clusters with their attributes, commands, and events
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MatterClusterInfo:
    """Information about a Matter cluster"""
    cluster_id: str
    cluster_name: str
    description: str
    category: str
    attributes: List[str]
    commands: List[str]
    events: List[str]

class MatterClusters:
    """Matter cluster definitions and management"""
    
    @staticmethod
    def get_all_clusters() -> Dict[str, MatterClusterInfo]:
        """Get all defined Matter clusters"""
        return {
            "on_off": MatterClusterInfo(
                cluster_id="0x0006",
                cluster_name="On/Off",
                description="Basic on/off functionality for devices",
                category="lighting",
                attributes=["OnOff", "GlobalSceneControl", "OnTime", "OffWaitTime"],
                commands=["Off", "On", "Toggle"],
                events=["StateChanged"]
            ),
            "level_control": MatterClusterInfo(
                cluster_id="0x0008", 
                cluster_name="Level Control",
                description="Controls the level/brightness of devices",
                category="lighting",
                attributes=["CurrentLevel", "RemainingTime", "MinLevel", "MaxLevel"],
                commands=["MoveToLevel", "Move", "Step", "Stop", "MoveToLevelWithOnOff"],
                events=["LevelChanged"]
            ),
            "door_lock": MatterClusterInfo(
                cluster_id="0x0101",
                cluster_name="Door Lock",
                description="Controls door lock functionality",
                category="access_control", 
                attributes=["LockState", "LockType", "ActuatorEnabled", "AutoRelockTime"],
                commands=["LockDoor", "UnlockDoor", "SetCredential", "ClearCredential"],
                events=["DoorLockAlarm", "LockOperation", "LockOperationError"]
            ),
            "thermostat": MatterClusterInfo(
                cluster_id="0x0201",
                cluster_name="Thermostat",
                description="Controls HVAC thermostat functionality", 
                category="hvac",
                attributes=["LocalTemperature", "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "SystemMode"],
                commands=["SetpointRaiseLower", "SetWeeklySchedule", "GetWeeklySchedule"],
                events=["TemperatureChanged", "SetpointChanged"]
            ),
            "window_covering": MatterClusterInfo(
                cluster_id="0x0102",
                cluster_name="Window Covering",
                description="Controls window covering devices like blinds and shades",
                category="covering",
                attributes=["Type", "PhysicalClosedLimitLift", "PhysicalClosedLimitTilt", "CurrentPositionLift"],
                commands=["UpOrOpen", "DownOrClose", "StopMotion", "GoToLiftValue", "GoToTiltValue"],
                events=["PositionChanged", "TargetChanged"]
            ),
            "color_control": MatterClusterInfo(
                cluster_id="0x0300",
                cluster_name="Color Control",
                description="Controls color properties of color-capable lights",
                category="lighting",
                attributes=["CurrentHue", "CurrentSaturation", "CurrentX", "CurrentY", "ColorTemperatureMireds"],
                commands=["MoveToHue", "MoveHue", "StepHue", "MoveToSaturation", "MoveToColor"],
                events=["ColorChanged", "HueChanged", "SaturationChanged"]
            ),
            "occupancy_sensing": MatterClusterInfo(
                cluster_id="0x0406",
                cluster_name="Occupancy Sensing",
                description="Detects occupancy in a space",
                category="sensing",
                attributes=["Occupancy", "OccupancySensorType", "PIROccupiedToUnoccupiedDelay"],
                commands=[],  # Server-only cluster
                events=["OccupancyChanged"]
            ),
            "temperature_measurement": MatterClusterInfo(
                cluster_id="0x0402",
                cluster_name="Temperature Measurement",
                description="Measures ambient temperature",
                category="sensing",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],  # Server-only cluster
                events=["TemperatureChanged"]
            )
        }
    
    @staticmethod
    def get_cluster(cluster_name: str) -> MatterClusterInfo:
        """Get specific cluster by name"""
        clusters = MatterClusters.get_all_clusters()
        if cluster_name not in clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(clusters.keys())}")
        return clusters[cluster_name]
    
    @staticmethod
    def get_cluster_names() -> List[str]:
        """Get list of all cluster names"""
        return list(MatterClusters.get_all_clusters().keys())
    
    @staticmethod
    def get_clusters_by_category(category: str) -> Dict[str, MatterClusterInfo]:
        """Get all clusters in a specific category"""
        all_clusters = MatterClusters.get_all_clusters()
        return {
            name: cluster for name, cluster in all_clusters.items() 
            if cluster.category == category
        }
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get list of all cluster categories"""
        clusters = MatterClusters.get_all_clusters()
        return list(set(cluster.category for cluster in clusters.values()))

# Matter-specific patterns for code filtering (following ProtocolGPT approach)
MATTER_PATTERNS = {
    "cluster": r'cluster|Cluster|CLUSTER',
    "state": r'state|State|STATE|status|Status',
    "command": r'command|Command|COMMAND|cmd|Cmd',
    "attribute": r'attribute|Attribute|ATTRIBUTE|attr|Attr',
    "event": r'event|Event|EVENT|callback|Callback'
}

# Configuration following ProtocolGPT defaults
PROTOCOLGPT_CONFIG = {
    "max_tokens": 4096,
    "chunk_size": 2056,
    "chunk_overlap": 256,
    "k": 4,
    "temperature": 0.2  # Low for factual FSM responses
}

# File extensions to analyze (following ProtocolGPT)
ALLOW_FILES = [
    '.txt', '.js', '.mjs', '.ts', '.tsx', '.css', '.scss', '.less', '.html', '.htm', 
    '.json', '.py', '.java', '.c', '.cpp', '.cs', '.go', '.php', '.rb', '.rs', 
    '.swift', '.kt', '.scala', '.m', '.h', '.sh', '.pl', '.pm', '.lua', '.sql'
]
