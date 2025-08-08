#!/usr/bin/env python3
"""
Matter Cluster-Specific FSM Generator
Following Les Modeling Framework from USENIX Security Papers
Creates formal logic models using State-Transitional Logic Rules and Event Generation Rules
Based on rag_implementation.py structure with LangGraph workflow
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Core LangChain/LangGraph imports (following rag_implementation.py)
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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

class MatterClusterLesFSMExtractor:
    """
    Cluster-specific FSM extractor following Les modeling framework from USENIX Security papers
    Generates formal logic models with State-Transitional Logic Rules and Event Generation Rules
    """
    
    def __init__(self, spec_path: str, model_name: str = "llama3.1"):
        self.spec_path = spec_path
        self.model_name = model_name
        self.vector_store = None
        self.graph = None
        self.memory = MemorySaver()
        
        # Matter clusters definition
        self.matter_clusters = self._define_matter_clusters()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (following rag_implementation.py pattern)
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_process_spec()
        self._setup_graph()
    
    def _define_matter_clusters(self) -> Dict[str, MatterClusterInfo]:
        """Define comprehensive Matter clusters based on specification"""
        return {
            # Base/Utility Clusters (0x0000-0x003F)
            "identify": MatterClusterInfo(
                cluster_id="0x0003",
                cluster_name="Identify",
                description="Provides an interface for identifying devices",
                category="base",
                attributes=["IdentifyTime", "IdentifyType"],
                commands=["Identify", "TriggerEffect"],
                events=[]
            ),
            "groups": MatterClusterInfo(
                cluster_id="0x0004",
                cluster_name="Groups",
                description="Manages group membership of devices",
                category="base",
                attributes=["NameSupport"],
                commands=["AddGroup", "ViewGroup", "GetGroupMembership", "RemoveGroup", "RemoveAllGroups", "AddGroupIfIdentifying"],
                events=[]
            ),
            "scenes": MatterClusterInfo(
                cluster_id="0x0005",
                cluster_name="Scenes",
                description="Manages scene functionality for groups of devices",
                category="base",
                attributes=["SceneCount", "CurrentScene", "CurrentGroup", "SceneValid", "NameSupport", "LastConfiguredBy"],
                commands=["AddScene", "ViewScene", "RemoveScene", "RemoveAllScenes", "StoreScene", "RecallScene", "GetSceneMembership"],
                events=[]
            ),
            "on_off": MatterClusterInfo(
                cluster_id="0x0006",
                cluster_name="On/Off",
                description="Basic on/off functionality for devices",
                category="lighting",
                attributes=["OnOff", "GlobalSceneControl", "OnTime", "OffWaitTime", "StartUpOnOff"],
                commands=["Off", "On", "Toggle", "OffWithEffect", "OnWithRecallGlobalScene", "OnWithTimedOff"],
                events=[]
            ),
            "level_control": MatterClusterInfo(
                cluster_id="0x0008", 
                cluster_name="Level Control",
                description="Controls the level/brightness of devices",
                category="lighting",
                attributes=["CurrentLevel", "RemainingTime", "MinLevel", "MaxLevel", "CurrentFrequency", "MinFrequency", "MaxFrequency", "Options", "OnOffTransitionTime", "OnLevel", "OnTransitionTime", "OffTransitionTime", "DefaultMoveRate", "StartUpCurrentLevel"],
                commands=["MoveToLevel", "Move", "Step", "Stop", "MoveToLevelWithOnOff", "MoveWithOnOff", "StepWithOnOff", "StopWithOnOff", "MoveToClosestFrequency"],
                events=[]
            ),
            "alarms": MatterClusterInfo(
                cluster_id="0x0009",
                cluster_name="Alarms",
                description="Manages device alarms",
                category="base",
                attributes=["AlarmCount"],
                commands=["ResetAlarm", "ResetAllAlarms", "GetAlarm", "ResetAlarmLog"],
                events=["Alarm"]
            ),
            "descriptor": MatterClusterInfo(
                cluster_id="0x001D",
                cluster_name="Descriptor",
                description="Provides device type and endpoint information",
                category="base",
                attributes=["DeviceTypeList", "ServerList", "ClientList", "PartsList"],
                commands=[],
                events=[]
            ),
            "binding": MatterClusterInfo(
                cluster_id="0x001E",
                cluster_name="Binding",
                description="Manages binding table for device-to-device communication",
                category="base",
                attributes=["Binding"],
                commands=[],
                events=[]
            ),
            "access_control": MatterClusterInfo(
                cluster_id="0x001F",
                cluster_name="Access Control",
                description="Manages access control lists and permissions",
                category="security",
                attributes=["Acl", "Extension", "SubjectsPerAccessControlEntry", "TargetsPerAccessControlEntry", "AccessControlEntriesPerFabric"],
                commands=[],
                events=["AccessControlEntryChanged", "AccessControlExtensionChanged"]
            ),
            
            # Application Clusters (0x0100+)
            "basic_information": MatterClusterInfo(
                cluster_id="0x0028",
                cluster_name="Basic Information",
                description="Provides basic device information",
                category="base",
                attributes=["DataModelRevision", "VendorName", "VendorID", "ProductName", "ProductID", "NodeLabel", "Location", "HardwareVersion", "HardwareVersionString", "SoftwareVersion", "SoftwareVersionString", "ManufacturingDate", "PartNumber", "ProductURL", "ProductLabel", "SerialNumber", "LocalConfigDisabled", "Reachable", "UniqueID", "CapabilityMinima", "SpecificationVersion", "MaxPathsPerInvoke"],
                commands=[],
                events=["StartUp", "ShutDown", "Leave", "ReachableChanged"]
            ),
            "door_lock": MatterClusterInfo(
                cluster_id="0x0101",
                cluster_name="Door Lock",
                description="Controls door lock functionality",
                category="closure", 
                attributes=["LockState", "LockType", "ActuatorEnabled", "DoorState", "DoorOpenEvents", "DoorClosedEvents", "OpenPeriod", "NumberOfTotalUsersSupported", "NumberOfPINUsersSupported", "NumberOfRFIDUsersSupported", "NumberOfWeekDaySchedulesSupportedPerUser", "NumberOfYearDaySchedulesSupportedPerUser", "NumberOfHolidaySchedulesSupported", "MaxPINCodeLength", "MinPINCodeLength", "MaxRFIDCodeLength", "MinRFIDCodeLength", "CredentialRulesSupport", "NumberOfCredentialsSupportedPerUser", "Language", "LEDSettings", "AutoRelockTime", "SoundVolume", "OperatingMode", "SupportedOperatingModes", "DefaultConfigurationRegister", "EnableLocalProgramming", "EnableOneTouchLocking", "EnableInsideStatusLED", "EnablePrivacyModeButton", "LocalProgrammingFeatures", "WrongCodeEntryLimit", "UserCodeTemporaryDisableTime", "SendPINOverTheAir", "RequirePINforRemoteOperation", "ExpiringUserTimeout"],
                commands=["LockDoor", "UnlockDoor", "UnlockWithTimeout", "SetWeekDaySchedule", "GetWeekDaySchedule", "ClearWeekDaySchedule", "SetYearDaySchedule", "GetYearDaySchedule", "ClearYearDaySchedule", "SetHolidaySchedule", "GetHolidaySchedule", "ClearHolidaySchedule", "SetUser", "GetUser", "ClearUser", "SetCredential", "GetCredential", "ClearCredential", "UnboltDoor"],
                events=["DoorLockAlarm", "DoorStateChange", "LockOperation", "LockOperationError", "LockUserChange"]
            ),
            "window_covering": MatterClusterInfo(
                cluster_id="0x0102",
                cluster_name="Window Covering",
                description="Controls window covering devices",
                category="closure",
                attributes=["Type", "PhysicalClosedLimitLift", "PhysicalClosedLimitTilt", "CurrentPositionLift", "CurrentPositionTilt", "NumberOfActuationsLift", "NumberOfActuationsTilt", "ConfigStatus", "CurrentPositionLiftPercentage", "CurrentPositionTiltPercentage", "OperationalStatus", "TargetPositionLiftPercent100ths", "TargetPositionTiltPercent100ths", "EndProductType", "CurrentPositionLiftPercent100ths", "CurrentPositionTiltPercent100ths", "InstalledOpenLimitLift", "InstalledClosedLimitLift", "InstalledOpenLimitTilt", "InstalledClosedLimitTilt", "Mode", "SafetyStatus"],
                commands=["UpOrOpen", "DownOrClose", "StopMotion", "GoToLiftValue", "GoToLiftPercentage", "GoToTiltValue", "GoToTiltPercentage"],
                events=[]
            ),
            "pump_configuration_control": MatterClusterInfo(
                cluster_id="0x0200",
                cluster_name="Pump Configuration and Control",
                description="Controls pump configuration and operation",
                category="hvac",
                attributes=["MaxPressure", "MaxSpeed", "MaxFlow", "MinConstPressure", "MaxConstPressure", "MinCompPressure", "MaxCompPressure", "MinConstSpeed", "MaxConstSpeed", "MinConstFlow", "MaxConstFlow", "MinConstTemp", "MaxConstTemp", "PumpStatus", "EffectiveOperationMode", "EffectiveControlMode", "Capacity", "Speed", "LifetimeRunningHours", "Power", "LifetimeEnergyConsumed", "OperationMode", "ControlMode", "AlarmMask"],
                commands=[],
                events=["SupplyVoltageLow", "SupplyVoltageHigh", "PowerMissingPhase", "SystemPressureLow", "SystemPressureHigh", "DryRunning", "MotorTemperatureHigh", "PumpMotorFatalFailure", "ElectronicTemperatureHigh", "PumpBlocked", "SensorFailure", "ElectronicNonFatalFailure", "ElectronicFatalFailure", "GeneralFault"]
            ),
            "thermostat": MatterClusterInfo(
                cluster_id="0x0201",
                cluster_name="Thermostat",
                description="Controls HVAC thermostat functionality", 
                category="hvac",
                attributes=["LocalTemperature", "OutdoorTemperature", "Occupancy", "AbsMinHeatSetpointLimit", "AbsMaxHeatSetpointLimit", "AbsMinCoolSetpointLimit", "AbsMaxCoolSetpointLimit", "PICoolingDemand", "PIHeatingDemand", "HVACSystemTypeConfiguration", "LocalTemperatureCalibration", "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "UnoccupiedCoolingSetpoint", "UnoccupiedHeatingSetpoint", "MinHeatSetpointLimit", "MaxHeatSetpointLimit", "MinCoolSetpointLimit", "MaxCoolSetpointLimit", "MinSetpointDeadBand", "RemoteSensing", "ControlSequenceOfOperation", "SystemMode", "ThermostatRunningMode", "StartOfWeek", "NumberOfWeeklyTransitions", "NumberOfDailyTransitions", "TemperatureSetpointHold", "TemperatureSetpointHoldDuration", "ThermostatProgrammingOperationMode", "ThermostatRunningState", "SetpointChangeSource", "SetpointChangeAmount", "SetpointChangeSourceTimestamp", "OccupiedSetback", "OccupiedSetbackMin", "OccupiedSetbackMax", "UnoccupiedSetback", "UnoccupiedSetbackMin", "UnoccupiedSetbackMax", "EmergencyHeatDelta", "ACType", "ACCapacity", "ACRefrigerantType", "ACCompressorType", "ACErrorCode", "ACLouverPosition", "ACCoilTemperature", "ACCapacityformat"],
                commands=["SetpointRaiseLower", "SetWeeklySchedule", "GetWeeklySchedule", "ClearWeeklySchedule", "GetRelayStatusLog"],
                events=[]
            ),
            "fan_control": MatterClusterInfo(
                cluster_id="0x0202",
                cluster_name="Fan Control",
                description="Controls fan speed and modes",
                category="hvac",
                attributes=["FanMode", "FanModeSequence", "PercentSetting", "PercentCurrent", "SpeedMax", "SpeedSetting", "SpeedCurrent", "RockSupport", "RockSetting", "WindSupport", "WindSetting", "AirflowDirection"],
                commands=[],
                events=[]
            ),
            "dehumidification_control": MatterClusterInfo(
                cluster_id="0x0203",
                cluster_name="Dehumidification Control", 
                description="Controls dehumidification functionality",
                category="hvac",
                attributes=["RelativeHumidity", "DehumidificationHysteresis", "DehumidificationMaxCool", "RelativeHumidityMode", "DehumidificationLockout", "DehumidificationHysteresisMax", "DehumidificationHysteresisMin"],
                commands=[],
                events=[]
            ),
            "thermostat_ui_config": MatterClusterInfo(
                cluster_id="0x0204",
                cluster_name="Thermostat UI Configuration",
                description="Configures thermostat user interface",
                category="hvac",
                attributes=["TemperatureDisplayMode", "KeypadLockout", "ScheduleProgrammingVisibility"],
                commands=[],
                events=[]
            ),
            "color_control": MatterClusterInfo(
                cluster_id="0x0300",
                cluster_name="Color Control",
                description="Controls color attributes of lighting devices",
                category="lighting",
                attributes=["CurrentHue", "CurrentSaturation", "RemainingTime", "CurrentX", "CurrentY", "DriftCompensation", "CompensationText", "ColorTemperatureMireds", "ColorMode", "Options", "NumberOfPrimaries", "Primary1X", "Primary1Y", "Primary1Intensity", "Primary2X", "Primary2Y", "Primary2Intensity", "Primary3X", "Primary3Y", "Primary3Intensity", "Primary4X", "Primary4Y", "Primary4Intensity", "Primary5X", "Primary5Y", "Primary5Intensity", "Primary6X", "Primary6Y", "Primary6Intensity", "WhitePointX", "WhitePointY", "ColorPointRX", "ColorPointRY", "ColorPointRIntensity", "ColorPointGX", "ColorPointGY", "ColorPointGIntensity", "ColorPointBX", "ColorPointBY", "ColorPointBIntensity", "EnhancedCurrentHue", "EnhancedColorMode", "ColorLoopActive", "ColorLoopDirection", "ColorLoopTime", "ColorLoopStartEnhancedHue", "ColorLoopStoredEnhancedHue", "ColorCapabilities", "ColorTempPhysicalMinMireds", "ColorTempPhysicalMaxMireds", "CoupleColorTempToLevelMinMireds", "StartUpColorTemperatureMireds"],
                commands=["MoveToHue", "MoveHue", "StepHue", "MoveToSaturation", "MoveSaturation", "StepSaturation", "MoveToHueAndSaturation", "MoveToColor", "MoveColor", "StepColor", "MoveToColorTemperature", "EnhancedMoveToHue", "EnhancedMoveHue", "EnhancedStepHue", "EnhancedMoveToHueAndSaturation", "ColorLoopSet", "StopMoveStep", "MoveColorTemperature", "StepColorTemperature"],
                events=[]
            ),
            "illuminance_measurement": MatterClusterInfo(
                cluster_id="0x0400",
                cluster_name="Illuminance Measurement",
                description="Measures ambient light illuminance",
                category="measurement",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance", "LightSensorType"],
                commands=[],
                events=[]
            ),
            "temperature_measurement": MatterClusterInfo(
                cluster_id="0x0402",
                cluster_name="Temperature Measurement",
                description="Measures ambient temperature",
                category="measurement",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],
                events=[]
            ),
            "pressure_measurement": MatterClusterInfo(
                cluster_id="0x0403",
                cluster_name="Pressure Measurement",
                description="Measures pressure",
                category="measurement",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance", "ScaledValue", "MinScaledValue", "MaxScaledValue", "ScaledTolerance", "Scale"],
                commands=[],
                events=[]
            ),
            "flow_measurement": MatterClusterInfo(
                cluster_id="0x0404",
                cluster_name="Flow Measurement",
                description="Measures flow rate",
                category="measurement",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],
                events=[]
            ),
            "relative_humidity_measurement": MatterClusterInfo(
                cluster_id="0x0405",
                cluster_name="Relative Humidity Measurement",
                description="Measures relative humidity",
                category="measurement",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],
                events=[]
            ),
            "occupancy_sensing": MatterClusterInfo(
                cluster_id="0x0406",
                cluster_name="Occupancy Sensing",
                description="Detects occupancy state",
                category="sensing",
                attributes=["Occupancy", "OccupancySensorType", "OccupancySensorTypeBitmap", "PIROccupiedToUnoccupiedDelay", "PIRUnoccupiedToOccupiedDelay", "PIRUnoccupiedToOccupiedThreshold", "UltrasonicOccupiedToUnoccupiedDelay", "UltrasonicUnoccupiedToOccupiedDelay", "UltrasonicUnoccupiedToOccupiedThreshold", "PhysicalContactOccupiedToUnoccupiedDelay", "PhysicalContactUnoccupiedToOccupiedDelay", "PhysicalContactUnoccupiedToOccupiedThreshold"],
                commands=[],
                events=[]
            ),
            "boolean_state": MatterClusterInfo(
                cluster_id="0x0045",
                cluster_name="Boolean State",
                description="Represents a boolean state",
                category="base",
                attributes=["StateValue"],
                commands=[],
                events=["StateChange"]
            ),
            "mode_select": MatterClusterInfo(
                cluster_id="0x0050",
                cluster_name="Mode Select",
                description="Allows selection from a list of supported modes",
                category="base",
                attributes=["Description", "StandardNamespace", "SupportedModes", "CurrentMode", "StartUpMode", "OnMode"],
                commands=["ChangeToMode"],
                events=[]
            ),
            "switch": MatterClusterInfo(
                cluster_id="0x003B",
                cluster_name="Switch",
                description="Reports switch position and state changes",
                category="base",
                attributes=["NumberOfPositions", "CurrentPosition", "MultiPressMax"],
                commands=[],
                events=["SwitchLatched", "InitialPress", "LongPress", "ShortRelease", "LongRelease", "MultiPressOngoing", "MultiPressComplete"]
            )
        }
    
    def _setup_llm(self):
        """Setup language model (following rag_implementation.py)"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,  # Low for consistent formal logic
            num_predict=4096
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model (following rag_implementation.py)"""
        print("🔤 Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _load_and_process_spec(self):
        """Load and process Matter specification document (following rag_implementation.py)"""
        print(f"📄 Loading document: {self.spec_path}")
        
        if not Path(self.spec_path).exists():
            raise FileNotFoundError(f"Document not found: {self.spec_path}")
        
        loader = UnstructuredHTMLLoader(self.spec_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        print("🗃️  Creating vector store...")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("✅ Vector store created and indexed")
    
    def _setup_graph(self):
        """Setup the LangGraph Les FSM extraction workflow"""
        print("🔗 Setting up Les FSM extraction workflow...")
        
        @tool(response_format="content_and_artifact")
        def retrieve_cluster_info(query: str):
            """Retrieve cluster-specific information from the Matter specification."""
            retrieved_docs = self.vector_store.similarity_search(query, k=6)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Matter Specification')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        self.retrieve_tool = retrieve_cluster_info
        
        graph_builder = StateGraph(MessagesState)
        
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            llm_with_tools = self.llm.bind_tools([retrieve_cluster_info])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        tools = ToolNode([retrieve_cluster_info])
        
        def generate_les_model(state: MessagesState):
            """Generate Les formal model using retrieved context."""
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are a formal verification specialist implementing modern FSM modeling standards "
                "for IoT protocols. Generate formal models using TLA+, Promela/SPIN, and temporal logic "
                "following industry best practices for protocol verification.\n\n"
                "MODELING STANDARDS:\n"
                "- Use TLA+ syntax for temporal logic and state machines\n"
                "- Follow Promela conventions for model checking with SPIN\n"
                "- Include LTL properties for safety and liveness\n"
                "- Model security properties and threat scenarios\n"
                "- Ensure finite state spaces for model checking\n\n"
                "CRITICAL: Output ONLY the formal model code without explanations, "
                "markdown formatting, or additional text. Follow the exact syntax requirements.\n\n"
                f"Matter Specification Context:\n{docs_content}"
            )
            
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_les_model", generate_les_model)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate_les_model")
        graph_builder.add_edge("generate_les_model", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("✅ Les FSM extraction workflow setup complete")

def get_les_cluster_queries(cluster: MatterClusterInfo) -> Dict[str, str]:
    """
    Generate formal model queries following modern FSM standards for IoT protocols
    Based on TLA+, Promela/SPIN, and formal verification best practices
    """
    
    base_context = f"""
CLUSTER: {cluster.cluster_name} (ID: {cluster.cluster_id})
CATEGORY: {cluster.category.upper()}
DESCRIPTION: {cluster.description}

ATTRIBUTES: {', '.join(cluster.attributes) if cluster.attributes else 'None'}
COMMANDS: {', '.join(cluster.commands) if cluster.commands else 'None'}
EVENTS: {', '.join(cluster.events) if cluster.events else 'None'}

TASK: Generate formal FSM model following IoT protocol verification standards.
"""

    return {
        "state_transition_rules": f"""{base_context}

Generate State Transition Rules using formal FSM syntax.

OUTPUT FORMAT (NO explanations):

*** Module: {cluster.cluster_name.replace(' ', '')}Cluster

*** Type definitions
DeviceState ::= {{idle, processing, active, error}}
UserRole ::= {{admin, user, guest}}
SecurityLevel ::= {{low, medium, high}}

*** State variables
VARIABLES
    device_state,     \* Current device state
    attributes,       \* Attribute values map
    user_session,     \* Current user session
    security_context  \* Security permissions

*** State predicates
Init == 
    /\\ device_state = "idle"
    /\\ attributes = [attribute |-> "default" : attribute \\in {{{', '.join(f'"{attr}"' for attr in cluster.attributes[:3]) if cluster.attributes else '"none"'}}}]
    /\\ user_session = [role |-> "guest", authenticated |-> FALSE]
    /\\ security_context = [level |-> "low"]

*** Transition predicates for each command
{chr(10).join([f'''
{cmd}Action(user) ==
    /\\ user_session.authenticated = TRUE
    /\\ device_state \\in {{"idle", "active"}}
    /\\ device_state' = "processing"
    /\\ UNCHANGED <<user_session, security_context>>
    /\\ attributes' = [attributes EXCEPT !["{cluster.attributes[0] if cluster.attributes else 'state'}"] = "{cmd.lower()}"]''' 
    for cmd in cluster.commands[:3]]) if cluster.commands else "NoCommands == TRUE"}

*** State invariants
StateInvariant == 
    /\\ device_state \\in DeviceState
    /\\ user_session.role \\in UserRole
    /\\ security_context.level \\in SecurityLevel

*** Temporal properties
Liveness == <>[]({cluster.cluster_name.replace(' ', '')}Ready)
Safety == []({cluster.cluster_name.replace(' ', '')}Safe)

Extract all state transitions for {cluster.cluster_name} cluster from Matter specification.
""",

        "security_properties": f"""{base_context}

Generate Security Properties for {cluster.cluster_name} cluster.

OUTPUT FORMAT (NO explanations):

*** Security Model for {cluster.cluster_name.replace(' ', '')}Cluster

*** Security states
SecurityState ::= {{unauthorized, authenticating, authorized, privileged}}
ThreatLevel ::= {{none, low, medium, high, critical}}

*** Security predicates
AuthenticationRequired ==
    \\A cmd \\in {{{', '.join(f'"{c}"' for c in cluster.commands[:5]) if cluster.commands else '"none"'}}} :
        (device_state = "processing" /\\ current_command = cmd)
        => user_session.authenticated = TRUE

AccessControlCheck ==
    \\A attr \\in {{{', '.join(f'"{a}"' for a in cluster.attributes[:5]) if cluster.attributes else '"none"'}}} :
        (attr_access_requested = attr)
        => security_context.level \\in {{"medium", "high"}}

*** Security violations
SV1_UnauthorizedAccess ==
    /\\ \\neg user_session.authenticated
    /\\ device_state = "processing"
    /\\ <>device_state = "active"  \* Eventually reaches active state

SV2_PrivilegeEscalation ==
    /\\ user_session.role = "guest"
    /\\ <>(user_session.role = "admin")
    /\\ \\neg authentication_upgrade_performed

SV3_DataIntegrityViolation ==
    /\\ \\E attr \\in attributes_domain :
        /\\ attr_modified(attr)
        /\\ \\neg authorized_modification(attr)

*** Threat model
ThreatModelInvariant ==
    /\\ (threat_level = "critical") => (security_context.level = "high")
    /\\ (threat_level = "high") => (user_session.role \\in {{"admin", "user"}})
    /\\ (device_state = "error") => <>recovery_possible

*** Security temporal properties
AuthenticationEventuallyRequired ==
    \\A operation \\in critical_operations :
        []<>(operation_requested => authentication_required)

Extract security model for {cluster.cluster_name} cluster following formal verification standards.
""",

        "formal_specification": f"""{base_context}

Generate Complete Formal Specification for {cluster.cluster_name} cluster.

OUTPUT FORMAT (NO explanations):

*** Formal Specification: {cluster.cluster_name.replace(' ', '')}Protocol

EXTENDS Naturals, Sequences, FiniteSets, TLC

*** Constants
CONSTANTS
    MAX_USERS,           \* Maximum concurrent users
    MAX_RETRY_ATTEMPTS,  \* Maximum command retry attempts  
    TIMEOUT_DURATION,    \* Command timeout in milliseconds
    CLUSTER_ID           \* Matter cluster identifier

*** Variables  
VARIABLES
    cluster_state,       \* Current cluster state
    attribute_values,    \* Map of attribute names to values
    pending_commands,    \* Queue of pending commands
    user_sessions,       \* Set of active user sessions
    event_history,       \* Sequence of generated events
    error_conditions,    \* Set of current error conditions
    network_status       \* Network connectivity state

*** Type definitions
ClusterState ::= {{"uninitialized", "idle", "processing", "active", "error", "shutdown"}}
CommandType ::= {{{', '.join(f'"{c}"' for c in cluster.commands[:8]) if cluster.commands else '"nop"'}}}
AttributeType ::= {{{', '.join(f'"{a}"' for a in cluster.attributes[:8]) if cluster.attributes else '"none"'}}}
EventType ::= {{{', '.join(f'"{e}"' for e in cluster.events[:5]) if cluster.events else '"none"'}}}

*** Initial state predicate
Init ==
    /\\ cluster_state = "uninitialized"
    /\\ attribute_values = [attr \\in AttributeType |-> "undefined"]
    /\\ pending_commands = <<>>
    /\\ user_sessions = {{}}
    /\\ event_history = <<>>
    /\\ error_conditions = {{}}
    /\\ network_status = "disconnected"

*** Action predicates
Initialize ==
    /\\ cluster_state = "uninitialized"
    /\\ cluster_state' = "idle"
    /\\ attribute_values' = [attr \\in AttributeType |-> "default"]
    /\\ network_status' = "connected"
    /\\ UNCHANGED <<pending_commands, user_sessions, event_history, error_conditions>>

ProcessCommand(cmd, user) ==
    /\\ cluster_state = "idle"
    /\\ cmd \\in CommandType
    /\\ user \\in user_sessions
    /\\ cluster_state' = "processing"
    /\\ pending_commands' = Append(pending_commands, [command |-> cmd, user |-> user, timestamp |-> Now])
    /\\ UNCHANGED <<attribute_values, user_sessions, event_history, error_conditions, network_status>>

*** Event generation predicates
GenerateEvent(event_type, data) ==
    /\\ event_type \\in EventType
    /\\ event_history' = Append(event_history, [type |-> event_type, data |-> data, timestamp |-> Now])
    /\\ UNCHANGED <<cluster_state, attribute_values, pending_commands, user_sessions, error_conditions, network_status>>

*** Safety properties
TypeInvariant ==
    /\\ cluster_state \\in ClusterState
    /\\ \\A attr \\in DOMAIN attribute_values : attr \\in AttributeType
    /\\ \\A session \\in user_sessions : ValidUserSession(session)
    /\\ Len(pending_commands) <= MAX_COMMANDS_QUEUE

SafetyInvariant ==
    /\\ (cluster_state = "processing") => (Len(pending_commands) > 0)
    /\\ (network_status = "disconnected") => (cluster_state \\in {{"error", "shutdown"}})
    /\\ \\A error \\in error_conditions : error.severity \\in {{"low", "medium", "high", "critical"}}

*** Liveness properties  
EventualProgress ==
    /\\ (cluster_state = "processing") ~> (cluster_state \\in {{"idle", "active", "error"}})
    /\\ (network_status = "disconnected") ~> <>network_status = "connected"

*** Next state relation
Next ==
    \\/ Initialize
    \\/ \\E cmd \\in CommandType, user \\in user_sessions : ProcessCommand(cmd, user)
    \\/ \\E event \\in EventType, data \\in EventData : GenerateEvent(event, data)
    \\/ HandleTimeout
    \\/ HandleError
    \\/ Shutdown

*** Specification
Spec == Init /\\ [][Next]_vars /\\ WF_vars(ProcessCommand) /\\ WF_vars(HandleTimeout)

*** Properties to verify
THEOREM Spec => []TypeInvariant
THEOREM Spec => []SafetyInvariant  
THEOREM Spec => EventualProgress

Generate complete formal specification for {cluster.cluster_name} cluster protocol.
""",

        "promela_model": f"""{base_context}

Generate Promela/SPIN model for {cluster.cluster_name} cluster verification.

OUTPUT FORMAT (NO explanations):

/* Promela Model: {cluster.cluster_name.replace(' ', '')}Cluster */

#define MAX_USERS 3
#define MAX_COMMANDS 10
#define MAX_ATTRIBUTES 8

/* Message types */
mtype = {{ 
    {', '.join(cluster.commands[:8]) if cluster.commands else 'nop'},
    {', '.join(cluster.events[:5]) if cluster.events else 'none'}, 
    ack, nack, timeout, error
}};

/* Cluster states */
mtype cluster_state = idle;
mtype states = {{ idle, processing, active, error, shutdown }};

/* Attribute values */
typedef AttributeMap {{
    {chr(10).join([f'    int {attr.lower()[:8]};' for attr in cluster.attributes[:8]]) if cluster.attributes else '    int dummy;'}
}};

AttributeMap attributes;

/* User session management */
typedef UserSession {{
    bool authenticated;
    byte role; /* 0=guest, 1=user, 2=admin */
    byte session_id;
}};

UserSession users[MAX_USERS];
byte active_users = 0;

/* Command queue */
typedef Command {{
    mtype cmd_type;
    byte user_id;
    byte timestamp;
}};

Command command_queue[MAX_COMMANDS];
byte queue_head = 0;
byte queue_tail = 0;

/* Security context */
typedef SecurityContext {{
    byte threat_level;  /* 0=none, 1=low, 2=medium, 3=high, 4=critical */
    bool access_granted;
    byte last_auth_time;
}};

SecurityContext security;

/* Network channels */
chan user_commands = [10] of {{ mtype, byte }};
chan cluster_events = [10] of {{ mtype, byte }};
chan security_alerts = [5] of {{ mtype, byte }};

/* Initialize cluster */
inline init_cluster() {{
    cluster_state = idle;
    {chr(10).join([f'    attributes.{attr.lower()[:8]} = 0;' for attr in cluster.attributes[:3]]) if cluster.attributes else '    attributes.dummy = 0;'}
    security.threat_level = 0;
    security.access_granted = false;
}}

/* Process user command */
inline process_command(cmd, user_id) {{
    atomic {{
        if
        :: (cluster_state == idle && users[user_id].authenticated) ->
            cluster_state = processing;
            if
            {chr(10).join([f'            :: (cmd == {cmd.lower()}) -> attributes.{cluster.attributes[i % len(cluster.attributes)].lower()[:8]}++;' 
                          for i, cmd in enumerate(cluster.commands[:5])]) if cluster.commands and cluster.attributes else '            :: (cmd == nop) -> skip;'}
            fi;
            cluster_state = active;
            cluster_events ! cmd, user_id;
        :: else -> 
            security_alerts ! error, user_id;
        fi
    }}
}}

/* User authentication process */
proctype UserAuth(byte uid) {{
    users[uid].session_id = uid;
    
    do
    :: atomic {{
        if
        :: (active_users < MAX_USERS) ->
            users[uid].authenticated = true;
            users[uid].role = 1; /* default user role */
            active_users++;
            break;
        :: else ->
            printf("Max users reached\\n");
            break;
        fi
    }}
    od;
    
    /* User activity loop */
    do
    :: user_commands ? cmd, _ ->
        process_command(cmd, uid);
    :: timeout ->
        users[uid].authenticated = false;
        active_users--;
        break;
    od
}}

/* Cluster state machine */
proctype ClusterStateMachine() {{
    mtype cmd;
    byte user_id;
    
    init_cluster();
    
    do
    :: user_commands ? cmd, user_id ->
        process_command(cmd, user_id);
    :: timeout ->
        if
        :: (cluster_state == processing) ->
            cluster_state = error;
            security_alerts ! timeout, 0;
        :: else -> skip;
        fi
    :: cluster_state == error ->
        cluster_state = idle;
        printf("Cluster recovered from error\\n");
    od
}}

/* Security monitor */
proctype SecurityMonitor() {{
    mtype alert_type;
    byte context;
    
    do
    :: security_alerts ? alert_type, context ->
        security.threat_level++;
        if
        :: (security.threat_level > 3) ->
            printf("CRITICAL SECURITY THREAT DETECTED\\n");
            security.access_granted = false;
        :: else -> skip;
        fi
    :: timeout ->
        if
        :: (security.threat_level > 0) ->
            security.threat_level--;
        :: else -> skip;
        fi
    od
}}

/* LTL Properties */
ltl safety1 {{ [](cluster_state == processing -> <>(cluster_state == active || cluster_state == error)) }}
ltl safety2 {{ []((active_users > 0) -> <>(active_users == 0)) }}
ltl security1 {{ [](security.threat_level == 4 -> <>security.threat_level < 4) }}
ltl liveness1 {{ []<>(cluster_state == idle) }}

/* Main process */
init {{
    atomic {{
        run ClusterStateMachine();
        run SecurityMonitor();
        run UserAuth(0);
        run UserAuth(1);
    }}
}}

Generate Promela model for {cluster.cluster_name} cluster suitable for SPIN model checker.
"""
    }

def extract_formal_cluster_model(extractor: MatterClusterLesFSMExtractor, cluster_name: str, 
                               query_type: str, thread_id: str = "default") -> Dict[str, Any]:
    """Extract formal model for specific cluster using modern FSM standards"""
    if cluster_name not in extractor.matter_clusters:
        raise ValueError(f"Unknown cluster: {cluster_name}")
    
    cluster = extractor.matter_clusters[cluster_name]
    queries = get_les_cluster_queries(cluster)
    
    if query_type not in queries:
        raise ValueError(f"Unknown query type: {query_type}")
    
    query = queries[query_type]
    
    print(f"\n🔍 Extracting {query_type} for {cluster.cluster_name} cluster")
    print("=" * 60)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤔 Processing with formal verification workflow...")
    final_response = None
    
    for step in extractor.graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config=config,
    ):
        final_message = step["messages"][-1]
        if hasattr(final_message, 'content') and final_message.content:
            if final_message.type == "ai" and not getattr(final_message, 'tool_calls', None):
                final_response = final_message.content
    
    if not final_response:
        raise ValueError("No response generated from formal verification workflow")
    
    # Clean up the response to extract only formal model code
    cleaned_response = clean_formal_model_output(final_response)
    
    print(f"\n🤖 Formal Model Generated")
    print("=" * 40)
    print(cleaned_response[:800] + "..." if len(cleaned_response) > 800 else cleaned_response)
    
    return {
        "cluster_name": cluster_name,
        "cluster_id": cluster.cluster_id,
        "query_type": query_type,
        "formal_model": cleaned_response,
        "raw_response": final_response,
        "thread_id": thread_id,
        "modeling_framework": "Modern_FSM_Standards"
    }

def clean_formal_model_output(response: str) -> str:
    """Clean up LLM response to extract only formal model code"""
    # Remove markdown code blocks
    response = re.sub(r'```[a-zA-Z]*\n?', '', response)
    response = re.sub(r'```', '', response)
    
    # Remove common explanatory text
    lines = response.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        'based on', 'following', 'analysis', 'here is', 'this model',
        'explanation', 'note:', 'important:', 'output:', 'format:', 'generated'
    ]
    
    for line in lines:
        line = line.strip()
        if line and not any(pattern in line.lower() for pattern in skip_patterns):
            # Keep formal model syntax lines
            if (line.startswith('***') or 
                line.startswith('VARIABLES') or 
                line.startswith('CONSTANTS') or
                line.startswith('EXTENDS') or
                line.startswith('Init ==') or
                line.startswith('Next ==') or
                line.startswith('Spec ==') or
                line.startswith('THEOREM') or
                line.startswith('/\\') or
                line.startswith('\\') or
                line.startswith('ltl ') or
                line.startswith('mtype') or
                line.startswith('proctype') or
                line.startswith('typedef') or
                line.startswith('inline') or
                line.startswith('atomic') or
                line.startswith('chan ') or
                line.startswith('#define') or
                line.startswith('/*') or
                line.startswith('init') or
                '::=' in line or
                '==' in line or
                '->' in line or
                '~>' in line or
                '/\\' in line or
                '\\/' in line or
                '[]' in line or
                '<>' in line):
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def main():
    """Main function for Matter Cluster Formal FSM Extraction"""
    print("🚀 Initializing Matter Cluster Formal FSM Extractor")
    print("="*50)
    
    spec_path = os.path.join(os.path.dirname(__file__), "Matter_Specification.html")
    
    if not os.path.exists(spec_path):
        print(f"❌ Matter specification not found at: {spec_path}")
        print("Please copy Matter_Specification.html to this directory")
        return
    
    try:
        extractor = MatterClusterLesFSMExtractor(
            spec_path=spec_path,
            model_name="llama3.1"
        )
        
        print("\n✅ Initialization complete!")
        print(f"\n📋 Available clusters ({len(extractor.matter_clusters)}):")
        for i, (cluster_key, cluster) in enumerate(extractor.matter_clusters.items(), 1):
            print(f"  {i}. {cluster.cluster_name} ({cluster.cluster_id}) - {cluster.category}")
        
        query_types = ["state_transition_rules", "security_properties", "formal_specification", "promela_model"]
        print(f"\n🔧 Available model types:")
        for i, query_type in enumerate(query_types, 1):
            print(f"  {i}. {query_type}")
        
        print("\n" + "="*60)
        print("🚀 Matter Cluster Formal FSM Extraction Assistant")
        print("="*60)
        print("Extract formal models following modern IoT protocol verification standards!")
        print("Supports: TLA+, Promela/SPIN, Temporal Logic, Security Properties")
        print("Commands:")
        print("  - 'cluster_name query_type' (e.g., 'on_off state_transition_rules')")
        print("  - 'all cluster_name' to extract all models for a cluster")
        print("  - 'complete' to extract formal specifications for all clusters")
        print("  - 'quit', 'exit' to exit")
        print("="*60)
        
        results = []
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'complete':
                    print("\n🚀 Extracting formal specifications for all clusters...")
                    for cluster_name in extractor.matter_clusters.keys():
                        try:
                            thread_id = f"complete_{cluster_name}"
                            result = extract_formal_cluster_model(extractor, cluster_name, "formal_specification", thread_id)
                            results.append(result)
                            
                            os.makedirs("formal_fsm_results", exist_ok=True)
                            filename = f"formal_fsm_results/{cluster_name}_formal_specification.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                            # Also save as .tla file
                            tla_filename = f"formal_fsm_results/{cluster_name}_specification.tla"
                            with open(tla_filename, 'w', encoding='utf-8') as f:
                                f.write(result["formal_model"])
                            print(f"💾 Saved TLA+ model: {tla_filename}")
                            
                        except Exception as e:
                            print(f"❌ Error extracting {cluster_name}: {e}")
                    continue
                
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[0] == 'all':
                        cluster_name = parts[1]
                        if cluster_name in extractor.matter_clusters:
                            print(f"\n🚀 Extracting all formal models for {cluster_name} cluster...")
                            for query_type in query_types:
                                try:
                                    thread_id = f"all_{cluster_name}_{query_type}"
                                    result = extract_formal_cluster_model(extractor, cluster_name, query_type, thread_id)
                                    results.append(result)
                                    
                                    os.makedirs("formal_fsm_results", exist_ok=True)
                                    filename = f"formal_fsm_results/{cluster_name}_{query_type}.json"
                                    with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    print(f"💾 Saved: {filename}")
                                    
                                    # Save with appropriate extension
                                    if query_type == "promela_model":
                                        ext_filename = f"formal_fsm_results/{cluster_name}_model.pml"
                                    elif query_type == "formal_specification":
                                        ext_filename = f"formal_fsm_results/{cluster_name}_spec.tla"
                                    else:
                                        ext_filename = f"formal_fsm_results/{cluster_name}_{query_type}.txt"
                                    
                                    with open(ext_filename, 'w', encoding='utf-8') as f:
                                        f.write(result["formal_model"])
                                    print(f"💾 Saved formal model: {ext_filename}")
                                    
                                except Exception as e:
                                    print(f"❌ Error in {query_type}: {e}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_name}")
                    else:
                        cluster_name, query_type = parts
                        if cluster_name in extractor.matter_clusters and query_type in query_types:
                            thread_id = f"{cluster_name}_{query_type}"
                            result = extract_formal_cluster_model(extractor, cluster_name, query_type, thread_id)
                            results.append(result)
                            
                            os.makedirs("formal_fsm_results", exist_ok=True)
                            filename = f"formal_fsm_results/{cluster_name}_{query_type}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                            # Save with appropriate extension
                            if query_type == "promela_model":
                                ext_filename = f"formal_fsm_results/{cluster_name}_model.pml"
                            elif query_type == "formal_specification":
                                ext_filename = f"formal_fsm_results/{cluster_name}_spec.tla"
                            else:
                                ext_filename = f"formal_fsm_results/{cluster_name}_{query_type}.txt"
                                
                            with open(ext_filename, 'w', encoding='utf-8') as f:
                                f.write(result["formal_model"])
                            print(f"💾 Saved formal model: {ext_filename}")
                        else:
                            print(f"❌ Invalid cluster or query type")
                            print("Available clusters:", list(extractor.matter_clusters.keys()))
                            print("Available query types:", query_types)
                else:
                    print("❌ Invalid command format")
                    print("Use: 'cluster_name query_type' or 'all cluster_name' or 'complete'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save all results
        if results:
            os.makedirs("formal_fsm_results", exist_ok=True)
            all_results_file = "formal_fsm_results/all_formal_fsm_results.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved all {len(results)} results to: {all_results_file}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please make sure 'Matter_Specification.html' is in the same directory.")
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is available: ollama pull llama3.1")
        print("3. Check if all dependencies are installed")
        print("4. Ensure Matter_Specification.html exists in the current directory")
        logging.exception("Initialization error")

if __name__ == "__main__":
    main()
