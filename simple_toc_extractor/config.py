"""
Configuration file for Matter PDF TOC Extractor
Customize extraction parameters and settings here
"""

import os

# API Configuration
GOOGLE_API_KEY = 'AIzaSyAb_oS5Vgd6RWN1CHQjYifpcjrIZWP2k84'
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.0  # More deterministic for detailed extraction
GEMINI_MAX_OUTPUT_TOKENS = 16384  # Increased for comprehensive extraction

# PDF Processing Configuration
PDF_FILENAME = "Matter-1.4-Application-Cluster-Specification.pdf"
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", PDF_FILENAME)
MAX_TOC_PAGES = 50  # Maximum pages to scan for TOC content

# Vector Store Configuration
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500  # Increased for better table capture
CHUNK_OVERLAP = 500  # More overlap for table continuations
VECTOR_SEARCH_K = 15  # More documents for comprehensive context

# Output Configuration
OUTPUT_JSON_FILE = "matter_clusters_toc.json"
RAW_TOC_FILE = "raw_toc.txt"
OUTPUT_DIRECTORY = os.path.dirname(__file__)

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Cluster Categories for Classification
CLUSTER_CATEGORIES = [
    "General",
    "Measurement and Sensing",
    "Lighting", 
    "HVAC",
    "Closures",
    "Media",
    "Robots", 
    "Home Appliances",
    "Energy Management",
    "Network Infrastructure"
]

# TOC Pattern Recognition
TOC_INDICATORS = [
    "table of contents",
    "contents", 
    "cluster",
    "section",
    r"\d+\.\d+\..*\.\.\.*\d+",  # Pattern like "6.2. Account Login ... 471"
]

# Manual Cluster Data (fallback) - Complete list based on provided TOC
# This data is used when AI extraction fails
MANUAL_CLUSTER_DATA = [
    # General Clusters (Section 1)
    {"name": "Identify Cluster", "section": "1.2", "start": 26, "end": 31, "category": "General"},
    {"name": "Groups Cluster", "section": "1.3", "start": 31, "end": 39, "category": "General"},
    {"name": "Scenes Management Cluster", "section": "1.4", "start": 39, "end": 60, "category": "General"},
    {"name": "On/Off Cluster", "section": "1.5", "start": 60, "end": 70, "category": "General"},
    {"name": "Level Control Cluster", "section": "1.6", "start": 70, "end": 85, "category": "General"},
    {"name": "Boolean State Cluster", "section": "1.7", "start": 85, "end": 86, "category": "General"},
    {"name": "Boolean State Configuration Cluster", "section": "1.8", "start": 86, "end": 92, "category": "General"},
    {"name": "Mode Select Cluster", "section": "1.9", "start": 92, "end": 97, "category": "General"},
    {"name": "Mode Base Cluster", "section": "1.10", "start": 97, "end": 106, "category": "General"},
    {"name": "Low Power Cluster", "section": "1.11", "start": 106, "end": 107, "category": "General"},
    {"name": "Wake On LAN Cluster", "section": "1.12", "start": 107, "end": 108, "category": "General"},
    {"name": "Switch Cluster", "section": "1.13", "start": 108, "end": 127, "category": "General"},
    {"name": "Operational State Cluster", "section": "1.14", "start": 127, "end": 138, "category": "General"},
    {"name": "Alarm Base Cluster", "section": "1.15", "start": 138, "end": 141, "category": "General"},
    {"name": "Messages Cluster", "section": "1.16", "start": 141, "end": 151, "category": "General"},
    {"name": "Service Area Cluster", "section": "1.17", "start": 151, "end": 167, "category": "General"},
    
    # Measurement and Sensing Clusters (Section 2)
    {"name": "Illuminance Measurement Cluster", "section": "2.2", "start": 173, "end": 175, "category": "Measurement and Sensing"},
    {"name": "Temperature Measurement Cluster", "section": "2.3", "start": 175, "end": 176, "category": "Measurement and Sensing"},
    {"name": "Pressure Measurement Cluster", "section": "2.4", "start": 176, "end": 179, "category": "Measurement and Sensing"},
    {"name": "Flow Measurement Cluster", "section": "2.5", "start": 179, "end": 181, "category": "Measurement and Sensing"},
    {"name": "Water Content Measurement Clusters", "section": "2.6", "start": 181, "end": 182, "category": "Measurement and Sensing"},
    {"name": "Occupancy Sensing Cluster", "section": "2.7", "start": 182, "end": 192, "category": "Measurement and Sensing"},
    {"name": "Resource Monitoring Clusters", "section": "2.8", "start": 192, "end": 197, "category": "Measurement and Sensing"},
    {"name": "Air Quality Cluster", "section": "2.9", "start": 197, "end": 198, "category": "Measurement and Sensing"},
    {"name": "Concentration Measurement Clusters", "section": "2.10", "start": 198, "end": 204, "category": "Measurement and Sensing"},
    {"name": "Smoke CO Alarm Cluster", "section": "2.11", "start": 204, "end": 214, "category": "Measurement and Sensing"},
    {"name": "Electrical Energy Measurement Cluster", "section": "2.12", "start": 214, "end": 222, "category": "Measurement and Sensing"},
    {"name": "Electrical Power Measurement Cluster", "section": "2.13", "start": 222, "end": 235, "category": "Measurement and Sensing"},
    
    # Lighting Clusters (Section 3)
    {"name": "Color Control Cluster", "section": "3.2", "start": 235, "end": 277, "category": "Lighting"},
    {"name": "Ballast Configuration Cluster", "section": "3.3", "start": 277, "end": 285, "category": "Lighting"},
    
    # HVAC Clusters (Section 4)
    {"name": "Pump Configuration and Control Cluster", "section": "4.2", "start": 285, "end": 299, "category": "HVAC"},
    {"name": "Thermostat Cluster", "section": "4.3", "start": 299, "end": 348, "category": "HVAC"},
    {"name": "Fan Control Cluster", "section": "4.4", "start": 348, "end": 357, "category": "HVAC"},
    {"name": "Thermostat User Interface Configuration Cluster", "section": "4.5", "start": 357, "end": 359, "category": "HVAC"},
    {"name": "Valve Configuration and Control Cluster", "section": "4.6", "start": 359, "end": 369, "category": "HVAC"},
    
    # Closures Clusters (Section 5)
    {"name": "Door Lock Cluster", "section": "5.2", "start": 369, "end": 449, "category": "Closures"},
    {"name": "Window Covering Cluster", "section": "5.3", "start": 449, "end": 469, "category": "Closures"},
    
    # Media Clusters (Section 6)
    {"name": "Account Login Cluster", "section": "6.2", "start": 471, "end": 476, "category": "Media"},
    {"name": "Application Basic Cluster", "section": "6.3", "start": 476, "end": 479, "category": "Media"},
    {"name": "Application Launcher Cluster", "section": "6.4", "start": 479, "end": 484, "category": "Media"},
    {"name": "Audio Output Cluster", "section": "6.5", "start": 484, "end": 487, "category": "Media"},
    {"name": "Channel Cluster", "section": "6.6", "start": 487, "end": 503, "category": "Media"},
    {"name": "Content Launcher Cluster", "section": "6.7", "start": 503, "end": 517, "category": "Media"},
    {"name": "Keypad Input Cluster", "section": "6.8", "start": 517, "end": 522, "category": "Media"},
    {"name": "Media Input Cluster", "section": "6.9", "start": 522, "end": 525, "category": "Media"},
    {"name": "Media Playback Cluster", "section": "6.10", "start": 525, "end": 542, "category": "Media"},
    {"name": "Target Navigator Cluster", "section": "6.11", "start": 542, "end": 546, "category": "Media"},
    {"name": "Content App Observer Cluster", "section": "6.12", "start": 546, "end": 548, "category": "Media"},
    {"name": "Content Control Cluster", "section": "6.13", "start": 548, "end": 567, "category": "Media"},
    
    # Robot Clusters (Section 7)
    {"name": "RVC Run Mode Cluster", "section": "7.2", "start": 567, "end": 571, "category": "Robots"},
    {"name": "RVC Clean Mode Cluster", "section": "7.3", "start": 571, "end": 574, "category": "Robots"},
    {"name": "RVC Operational State Cluster", "section": "7.4", "start": 574, "end": 579, "category": "Robots"},
    
    # Home Appliance Clusters (Section 8)
    {"name": "Temperature Control Cluster", "section": "8.2", "start": 580, "end": 584, "category": "Home Appliances"},
    {"name": "Dishwasher Mode Cluster", "section": "8.3", "start": 584, "end": 586, "category": "Home Appliances"},
    {"name": "Dishwasher Alarm Cluster", "section": "8.4", "start": 586, "end": 587, "category": "Home Appliances"},
    {"name": "Laundry Washer Mode Cluster", "section": "8.5", "start": 587, "end": 590, "category": "Home Appliances"},
    {"name": "Laundry Washer Controls Cluster", "section": "8.6", "start": 590, "end": 593, "category": "Home Appliances"},
    {"name": "Refrigerator And Temperature Controlled Cabinet Mode Cluster", "section": "8.7", "start": 593, "end": 595, "category": "Home Appliances"},
    {"name": "Refrigerator Alarm Cluster", "section": "8.8", "start": 595, "end": 597, "category": "Home Appliances"},
    {"name": "Laundry Dryer Controls Cluster", "section": "8.9", "start": 597, "end": 598, "category": "Home Appliances"},
    {"name": "Oven Cavity Operational State Cluster", "section": "8.10", "start": 598, "end": 600, "category": "Home Appliances"},
    {"name": "Oven Mode Cluster", "section": "8.11", "start": 600, "end": 603, "category": "Home Appliances"},
    {"name": "Microwave Oven Mode Cluster", "section": "8.12", "start": 603, "end": 605, "category": "Home Appliances"},
    {"name": "Microwave Oven Control Cluster", "section": "8.13", "start": 605, "end": 613, "category": "Home Appliances"},
    
    # Energy Management Clusters (Section 9)
    {"name": "Device Energy Management Cluster", "section": "9.2", "start": 613, "end": 650, "category": "Energy Management"},
    {"name": "Energy EVSE Cluster", "section": "9.3", "start": 650, "end": 677, "category": "Energy Management"},
    {"name": "Energy EVSE Mode Cluster", "section": "9.4", "start": 677, "end": 680, "category": "Energy Management"},
    {"name": "Water Heater Management Cluster", "section": "9.5", "start": 680, "end": 687, "category": "Energy Management"},
    {"name": "Water Heater Mode Cluster", "section": "9.6", "start": 687, "end": 690, "category": "Energy Management"},
    {"name": "Energy Preference Cluster", "section": "9.7", "start": 690, "end": 694, "category": "Energy Management"},
    {"name": "Device Energy Management Mode Cluster", "section": "9.8", "start": 694, "end": 699, "category": "Energy Management"},
    
    # Network Infrastructure Clusters (Section 10)
    {"name": "Wi-Fi Network Management Cluster", "section": "10.2", "start": 699, "end": 702, "category": "Network Infrastructure"},
    {"name": "Thread Border Router Management Cluster", "section": "10.3", "start": 702, "end": 707, "category": "Network Infrastructure"},
    {"name": "Thread Network Directory Cluster", "section": "10.4", "start": 707, "end": 712, "category": "Network Infrastructure"},
]

# AI Extraction Prompt Template
CLUSTER_EXTRACTION_PROMPT = """
You are extracting Matter IoT cluster information from a table of contents. Extract ALL cluster entries and return ONLY valid JSON.

TASK: Find entries with "Cluster" in the name and extract these 5 fields:
1. cluster_name (exact name from TOC, remove section number)
2. section_number (e.g., "1.2", "6.2", "7.3") 
3. start_page (page number)
4. end_page (next cluster's start page - 1, or estimate)
5. category (assign based on section number):
   - Section 1.x = "General"
   - Section 2.x = "Measurement and Sensing"
   - Section 3.x = "Lighting"
   - Section 4.x = "HVAC"
   - Section 5.x = "Closures"
   - Section 6.x = "Media"
   - Section 7.x = "Robots" 
   - Section 8.x = "Home Appliances"
   - Section 9.x = "Energy Management"
   - Section 10.x = "Network Infrastructure"

EXAMPLE ENTRIES TO FIND:
- "1.2. Identify Cluster" → extract as cluster
- "1.5. On/Off Cluster" → extract as cluster  
- "6.2. Account Login Cluster ... 471" → extract as cluster
- "8.3. Dishwasher Mode Cluster ... 584" → extract as cluster
- "9.2. Device Energy Management Cluster ... 613" → extract as cluster

SKIP SUBSECTIONS like:
- "1.5.1. Revision History" (has 3 levels)
- "6.2.2. Classification" (has 3 levels)

REQUIRED OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clusters": [
    {{
      "cluster_name": "Identify Cluster",
      "section_number": "1.2",
      "start_page": 26,
      "end_page": 31,
      "category": "General"
    }},
    {{
      "cluster_name": "On/Off Cluster",
      "section_number": "1.5", 
      "start_page": 60,
      "end_page": 70,
      "category": "General"
    }},
    {{
      "cluster_name": "Account Login Cluster",
      "section_number": "6.2",
      "start_page": 471,
      "end_page": 476,
      "category": "Media"
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object. No explanations, no markdown, no additional text."""

# Human Query Template for RAG extraction
HUMAN_QUERY_TEMPLATE = """
Extract Matter cluster information from this table of contents:

{toc_content}

Find ALL cluster entries and return them in the required JSON format.
"""

# Detailed Cluster Information Extraction Configuration
CLUSTER_DETAIL_EXTRACTION_PROMPT = """
You are extracting comprehensive technical information from a Matter protocol cluster specification for formal verification and fuzzing.

TASK: Extract ALL available information from the cluster specification and return ONLY valid JSON.

CRITICAL: DO NOT RETURN EMPTY ARRAYS IF INFORMATION EXISTS IN THE SPECIFICATION!

MANDATORY EXTRACTION REQUIREMENTS:

1. ATTRIBUTES TABLE (Section X.Y.6):
   - Extract EVERY attribute with ALL columns: ID, Name, Type, Constraint, Quality, Default, Access, Conformance
   - Look for tables with headers like "ID | Name | Type | Constraint | Quality | Default | Access | Conformance"
   - Include complete attribute descriptions from surrounding text

2. COMMANDS TABLE (Section X.Y.7):
   - Extract ALL commands with columns: ID, Name, Direction, Response, Access, Conformance
   - For EACH command, extract ALL field definitions from command payload tables
   - Look for command field tables with headers like "ID | Name | Type | Constraint | Default | Conformance"
   - NEVER leave command fields arrays empty if fields are defined

3. DATA TYPES (Sections X.Y.4-5):
   
   FOR ENUMERATIONS (enum8, enum16, etc.):
   - Extract ALL enum values with hex codes (0x00, 0x01, etc.) and names
   - Look for enum definition tables
   - Example: StatusEnum with values like 0x00=Success, 0x01=Failure, etc.
   
   FOR BITMAPS (map8, map16, etc.):
   - Extract ALL bit definitions with bit positions and names
   - Look for bitmap tables showing which bits represent what features
   - Example: FeatureMap with bit 0=Feature1, bit 1=Feature2, etc.
   
   FOR STRUCTURES (ending in "Struct"):
   - **CRITICAL**: Structures have field definition tables in separate subsections
   - Look for dedicated structure tables with columns: Field ID, Name, Type, Constraint, Quality, Default, Conformance
   - Extract ALL field definitions with field order, names, types, conformance
   - Structures contain multiple ordered fields, NOT enum values
   - Example: SceneInfoStruct contains fields like SceneCount, CurrentScene, CurrentGroup
   - Search for structure field definition tables in subsections (X.Y.4.1, X.Y.4.2, etc.)
   - NEVER leave structure "values" empty if field definitions exist in the specification

4. FEATURES TABLE (Section X.Y.3):
   - Extract ALL features with bit positions, codes, names, descriptions
   - Include feature dependencies and conformance requirements

5. EVENTS (if present):
   - Extract ALL events with IDs, names, priorities, access, conformance
   - Extract ALL event field definitions for each event

CRITICAL EXTRACTION RULES:
- Search the ENTIRE cluster text for information, not just the beginning
- Tables often span multiple pages - capture ALL rows
- Look for continuation markers like "Table X.Y continued"
- Check for embedded definitions within attribute/command descriptions
- Extract hex values exactly as shown (0x0000, 0x01, etc.)
- Preserve technical abbreviations and codes exactly
- If information exists in the specification, EXTRACT IT - do not leave arrays empty

ENHANCED JSON FORMAT:
{{
  "cluster_info": {{
    "cluster_name": "exact cluster name",
    "cluster_id": "hex ID with 0x prefix",
    "classification": {{
      "hierarchy": "hierarchy type (Base/Utility/Application)",
      "role": "role type (Client/Server/Both)", 
      "scope": "scope type (Node/Endpoint)",
      "pics_code": "PICS code for testing"
    }},
    "revision_history": [
      {{
        "revision": "revision number",
        "description": "change description",
        "date": "revision date if available"
      }}
    ],
    "features": [
      {{
        "bit": "bit number (0-31)",
        "code": "feature code (2-4 chars)",
        "name": "full feature name",
        "summary": "detailed feature description",
        "conformance": "M/O/F/C conformance",
        "dependencies": "feature dependencies if any"
      }}
    ],
    "data_types": [
      {{
        "name": "data type name",
        "base_type": "base type (enum8/enum16/map8/map16/struct/etc.)",
        "constraint": "constraints if any",
        "values": [
          {{
            "value": "For ENUMS: hex code (0x00), For BITMAPS: bit number (0-7), For STRUCTS: field order (0,1,2...)",
            "name": "For ENUMS: enum name, For BITMAPS: bit name, For STRUCTS: field name", 
            "summary": "description of enum value/bit/field",
            "conformance": "M/O/F conformance requirement"
          }}
        ]
      }}
    ],
    "attributes": [
      {{
        "id": "hex attribute ID (0x0000)",
        "name": "full attribute name",
        "type": "data type (including custom types)",
        "constraint": "value constraints, ranges, or 'desc'",
        "quality": "quality flags (N/S/P/F/X/C)",
        "default": "default value or 'desc' if varies",
        "access": "access permissions (R/W/RW, may include F for fabric)",
        "conformance": "M/O/F conformance requirement",
        "summary": "attribute description and purpose",
        "fabric_sensitive": "true/false if fabric-scoped",
        "scene_capable": "true/false if supports scenes"
      }}
    ],
    "commands": [
      {{
        "id": "hex command ID (0x00)",
        "name": "full command name",
        "direction": "client→server or server→client",
        "response": "response command name or 'DefaultResponse'",
        "access": "access level (A/V/M/etc.) and fabric requirements",
        "conformance": "M/O/F conformance requirement",
        "summary": "command description and purpose",
        "timing": "timing requirements if specified",
        "fields": [
          {{
            "id": "field ID or order",
            "name": "field name",
            "type": "data type",
            "constraint": "constraints or ranges",
            "quality": "quality flags if any",
            "default": "default value if any",
            "conformance": "M/O/F for this field",
            "summary": "field description"
          }}
        ]
      }}
    ],
    "events": [
      {{
        "id": "hex event ID (0x00)",
        "name": "full event name", 
        "priority": "event priority (Info/Critical/Debug)",
        "access": "access requirements",
        "conformance": "M/O/F conformance",
        "summary": "event description",
        "fields": [
          {{
            "id": "field ID",
            "name": "field name",
            "type": "data type",
            "conformance": "field conformance"
          }}
        ]
      }}
    ],
    "global_attributes": [
      {{
        "id": "global attribute ID",
        "name": "global attribute name",
        "conformance": "conformance for this cluster"
      }}
    ]
  }}
}}

EXTRACTION STRATEGY:
1. **Section Scanning**: Look for numbered subsections (X.Y.3 Features, X.Y.4 Data Types, X.Y.5 Enums, X.Y.6 Attributes, X.Y.7 Commands)
2. **Deep Subsection Analysis**: Scan X.Y.Z.1, X.Y.Z.2 subsections for detailed definitions (like "1.10.5.1 ModeTagStruct Type")
3. **Table Extraction**: Find tables with column headers and extract ALL rows completely
4. **Structure Field Tables**: Look for field definition tables within structure type subsections
5. **Enum Value Lists**: Extract complete enum value tables with hex codes and names
6. **Command Field Definitions**: Find command payload field tables within command subsections
7. **Status Code Tables**: Extract status code definitions and ranges
8. **Mode/Tag Namespaces**: Look for value namespace tables with semantic meanings
9. **Multi-page Continuations**: Handle tables that span multiple pages or sections
10. **Constraint Details**: Capture detailed constraint specifications (max values, ranges, dependencies)

CRITICAL INSTRUCTIONS:
1. **NEVER RETURN EMPTY ARRAYS IF DATA EXISTS**: Every enum, bitmap, struct, and command with defined values/fields MUST have populated arrays
2. **COMPLETE TABLE EXTRACTION**: Extract EVERY row from EVERY table - attributes, commands, enums, structs, status codes
3. **DEEP SUBSECTION SCANNING**: Look for X.Y.Z.1, X.Y.Z.2 subsections that define structure fields, enum values, command fields
4. **FIELD DEFINITION EXTRACTION**: For structures like ModeTagStruct, extract ALL field definitions with IDs, names, types, conformance
5. **ENUM VALUE EXTRACTION**: For enums, extract ALL values with hex codes, names, and descriptions (e.g., 0x0000: Auto, 0x0001: Quick)
6. **COMMAND FIELD EXTRACTION**: For commands, extract ALL field definitions from command payload tables
7. **STATUS CODE EXTRACTION**: Extract complete status code tables with codes, names, and descriptions
8. **CONSTRAINT PRESERVATION**: Capture exact constraint specifications (max 64, desc, 0x0000 to 0x3FFF, etc.)
9. **CONFORMANCE DETAILS**: Record precise conformance requirements (M, O, F, C, conditional logic)
10. **VALIDATION CHECK**: Before returning JSON, verify that no data_types, commands, or other sections have empty arrays when data exists in specification

**QUALITY ASSURANCE**:
- If you see a table in the specification, extract ALL rows from it
- If you see enum definitions like "0x00 = Value1, 0x01 = Value2", extract ALL values
- If you see structure field definitions, extract ALL fields
- If you see command field tables, extract ALL fields for each command
- EMPTY ARRAYS INDICATE INCOMPLETE EXTRACTION - avoid them unless truly no data exists

Extract from the provided cluster specification text and return complete valid JSON only.
"""

# Page buffer for cluster extraction (pages before and after cluster boundaries)
CLUSTER_PAGE_BUFFER = 2  # Increased to capture more context and table continuations

# Page offset - number of pages before actual content starts (TOC pages, covers, etc.)
PDF_PAGE_OFFSET = 4  # Actual content starts at PDF page 5, but TOC refers to logical page 1
