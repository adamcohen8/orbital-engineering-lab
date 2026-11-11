"""Generate a loopback-only Yamcs mission database and least-privilege accounts."""

from __future__ import annotations

import json
import secrets
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from .contracts import NUMERIC_FIELDS, PARAMETERS
from .temperature_contracts import THERMAL_NUMERIC, THERMAL_PARAMETERS

YAMCS_VERSION = "5.13.6"
MAVEN_POM = f"""<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion><groupId>oel.local</groupId><artifactId>operations-runtime</artifactId><version>1</version><dependencies><dependency><groupId>org.yamcs</groupId><artifactId>yamcs-core</artifactId><version>{YAMCS_VERSION}</version></dependency><dependency><groupId>org.yamcs</groupId><artifactId>yamcs-web</artifactId><version>{YAMCS_VERSION}</version></dependency></dependencies></project>"""


def mission_database():
    root = ET.Element("SpaceSystem", name="oel", xmlns="http://www.omg.org/spec/XTCE/20180204")
    tm = ET.SubElement(root, "TelemetryMetaData")
    types = ET.SubElement(tm, "ParameterTypeSet")
    ET.SubElement(types, "FloatParameterType", name="number", sizeInBits="64")
    ET.SubElement(types, "StringParameterType", name="text")
    params = ET.SubElement(tm, "ParameterSet")
    for name in PARAMETERS + THERMAL_PARAMETERS:
        p = ET.SubElement(
            params,
            "Parameter",
            name=name,
            parameterTypeRef="number" if name in NUMERIC_FIELDS + THERMAL_NUMERIC else "text",
        )
        ET.SubElement(p, "ParameterProperties", dataSource="telemetered")
    cm = ET.SubElement(root, "CommandMetaData")
    types = ET.SubElement(cm, "ArgumentTypeSet")
    uint = ET.SubElement(types, "IntegerArgumentType", name="uint32", signed="false")
    ET.SubElement(
        uint, "IntegerDataEncoding", sizeInBits="32", encoding="unsigned", byteOrder="mostSignificantByteFirst"
    )
    number = ET.SubElement(types, "FloatArgumentType", name="double", sizeInBits="64")
    ET.SubElement(
        number, "FloatDataEncoding", sizeInBits="64", encoding="IEEE754_1985", byteOrder="mostSignificantByteFirst"
    )
    commands = ET.SubElement(cm, "MetaCommandSet")
    command = ET.SubElement(
        commands,
        "MetaCommand",
        name="PointTarget",
        shortDescription="Approval-bound ECI target pointing; endpoint enforces OEL forecast and command lease",
    )
    args = ET.SubElement(command, "ArgumentList")
    names = ("request_id", "ticket", "target_x", "target_y", "target_z", "hold_s")
    for name in names:
        ET.SubElement(args, "Argument", name=name, argumentTypeRef="uint32" if name in names[:2] else "double")
    entries = ET.SubElement(ET.SubElement(command, "CommandContainer", name="PointTarget"), "EntryList")
    for name in names:
        ET.SubElement(entries, "ArgumentRefEntry", argumentRef=name)
    command = ET.SubElement(
        commands,
        "MetaCommand",
        name="ThermalProcedure",
        shortDescription="Owner-configured suspend/resume procedure; no temperature prediction",
    )
    args = ET.SubElement(command, "ArgumentList")
    names = ("request_id", "ticket", "component_code", "operation_code")
    entries = ET.SubElement(ET.SubElement(command, "CommandContainer", name="ThermalProcedure"), "EntryList")
    for name in names:
        ET.SubElement(args, "Argument", name=name, argumentTypeRef="uint32")
        ET.SubElement(entries, "ArgumentRefEntry", argumentRef=name)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def configure(root: Path, *, http_port=18090, tm_port=11916, tc_port=11925):
    root.mkdir(parents=True, exist_ok=True)
    etc, mdb = root / "etc", root / "mdb"
    etc.mkdir(exist_ok=True)
    mdb.mkdir(exist_ok=True)
    credentials_path = root / "credentials.json"
    if credentials_path.exists():
        credentials = json.loads(credentials_path.read_text())
    else:
        credentials = {name: secrets.token_urlsafe(24) for name in ("agent", "endpoint", "viewer")}
        credentials_path.write_text(json.dumps(credentials))
        credentials_path.chmod(0o600)
    if "viewer" not in credentials:
        credentials["viewer"] = secrets.token_urlsafe(24)
        credentials_path.write_text(json.dumps(credentials))
    files = {
        "yamcs.yaml": {
            "services": [{"class": "org.yamcs.http.HttpServer", "args": {"address": "127.0.0.1", "port": http_port}}],
            "dataDir": "data",
            "instances": ["oel"],
            "secretKey": secrets.token_urlsafe(32),
        },
        "yamcs.oel.yaml": {
            "services": [
                {"class": "org.yamcs.ProcessorCreatorService", "args": {"name": "realtime", "type": "realtime"}},
                {"class": "org.yamcs.archive.CommandHistoryRecorder"},
                {"class": "org.yamcs.archive.ParameterRecorder"},
                {"class": "org.yamcs.plists.ParameterListService"},
            ],
            "dataLinks": [
                {
                    "name": "oel-telemetry",
                    "class": "org.yamcs.tctm.UdpParameterDataLink",
                    "stream": "pp_realtime",
                    "port": tm_port,
                    "json": True,
                },
                {
                    "name": "oel-command",
                    "class": "org.yamcs.tctm.UdpTcDataLink",
                    "stream": "tc_realtime",
                    "host": "127.0.0.1",
                    "port": tc_port,
                    "commandPostprocessorClassName": "org.yamcs.tctm.GenericCommandPostprocessor",
                },
            ],
            "mdb": [{"type": "xtce", "args": {"file": "mdb/xtce.xml"}}],
            "streamConfig": {
                "param": ["pp_realtime"],
                "event": ["events_realtime"],
                "cmdHist": ["cmdhist_realtime"],
                "tc": [{"name": "tc_realtime", "processor": "realtime"}],
            },
        },
        "processor.yaml": {
            "realtime": {
                "services": [
                    {"class": "org.yamcs.StreamTcCommandReleaser"},
                    {"class": "org.yamcs.tctm.StreamParameterProvider", "args": {"streams": ["pp_realtime"]}},
                ],
                "config": {"subscribeAll": True},
            }
        },
        "security.yaml": {
            "enabled": True,
            "authModules": [{"class": "org.yamcs.security.YamlAuthModule", "args": {"required": True}}],
        },
        "users.yaml": {
            "agent": {"password": credentials["agent"], "roles": ["OperationsAgent"]},
            "endpoint": {"password": credentials["endpoint"], "roles": ["Endpoint"]},
            "viewer": {"password": credentials["viewer"], "roles": ["ConsoleViewer"]},
        },
        "roles.yaml": {
            "ConsoleViewer": {
                "ReadParameter": ["/oel/.*"],
                "CommandHistory": ["/?oel/(PointTarget|ThermalProcedure)"],
                "System": ["GetMissionDatabase"],
            },
            "OperationsAgent": {
                "ReadParameter": ["/oel/.*"],
                "Command": ["/oel/PointTarget", "/oel/ThermalProcedure"],
                "CommandHistory": ["/?oel/(PointTarget|ThermalProcedure)"],
                "System": ["GetMissionDatabase"],
            },
            "Endpoint": {
                "ReadParameter": ["/oel/.*"],
                "CommandHistory": ["/?oel/(PointTarget|ThermalProcedure)"],
                "System": ["ModifyCommandHistory", "GetMissionDatabase"],
            },
        },
    }
    for name, value in files.items():
        path = etc / name
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        path.chmod(0o600)
    (mdb / "xtce.xml").write_text(mission_database())
    return credentials_path
