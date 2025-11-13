"""
SECS/GEM Equipment Interface for CVD Platform
Implements SECS-II/GEM (SEMI E4/E5/E30) communication protocol.
Provides standardized interface to semiconductor equipment.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


class SECSMessageType(Enum):
    """SECS-II Message Types (Stream, Function)"""
    # Equipment Status
    S1F1 = (1, 1)  # Are You There Request
    S1F2 = (1, 2)  # On Line Data
    S1F3 = (1, 3)  # Selected Equipment Status Request
    S1F4 = (1, 4)  # Selected Equipment Status Data

    # Equipment Control
    S2F15 = (2, 15)  # New Equipment Constant Send
    S2F16 = (2, 16)  # New Equipment Constant Acknowledge
    S2F41 = (2, 41)  # Host Command Send
    S2F42 = (2, 42)  # Host Command Acknowledge

    # Data Collection
    S6F11 = (6, 11)  # Event Report Send
    S6F12 = (6, 12)  # Event Report Acknowledge
    S6F19 = (6, 19)  # Individual Report Request
    S6F20 = (6, 20)  # Individual Report Data

    # Process Program Management
    S7F1 = (7, 1)  # Process Program Load Inquire
    S7F2 = (7, 2)  # Process Program Load Grant
    S7F3 = (7, 3)  # Process Program Send
    S7F4 = (7, 4)  # Process Program Acknowledge

    # Alarm Management
    S5F1 = (5, 1)  # Alarm Report Send
    S5F2 = (5, 2)  # Alarm Report Acknowledge


class EquipmentState(Enum):
    """GEM Equipment States (E30)"""
    EQUIPMENT_OFFLINE = 1
    ATTEMPT_ONLINE = 2
    HOST_OFFLINE = 3
    ONLINE_LOCAL = 4
    ONLINE_REMOTE = 5


class ProcessState(Enum):
    """Process States"""
    INIT = 1
    IDLE = 2
    SETUP = 3
    READY = 4
    EXECUTING = 5
    PAUSE = 6
    COMPLETE = 7


class AlarmSeverity(Enum):
    """Alarm Severity Levels"""
    WARNING = 1
    ALARM = 2
    CRITICAL = 3
    FAULT = 4


@dataclass
class SECSMessage:
    """SECS-II Message Structure"""
    stream: int
    function: int
    wait_bit: bool
    device_id: int
    system_bytes: int
    data: Any
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "function": self.function,
            "wait_bit": self.wait_bit,
            "device_id": self.device_id,
            "system_bytes": self.system_bytes,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EquipmentConstant:
    """Equipment Constant (EC) Definition"""
    ec_id: str
    name: str
    value: Any
    unit: str
    min_value: Optional[float]
    max_value: Optional[float]
    description: str


@dataclass
class DataVariable:
    """Status Variable (SV) or Data Variable (DV)"""
    var_id: str
    name: str
    value: Any
    unit: str
    timestamp: datetime


@dataclass
class AlarmData:
    """Alarm Information"""
    alarm_id: str
    alarm_code: int
    alarm_text: str
    severity: AlarmSeverity
    timestamp: datetime
    equipment_id: str
    state_change: bool


class SECSGEMInterface:
    """
    SECS/GEM Interface Implementation
    Handles communication between CVD equipment and host (MES/ERP)
    """

    def __init__(self, equipment_id: str, host_ip: str, port: int):
        self.equipment_id = equipment_id
        self.host_ip = host_ip
        self.port = port
        self.device_id = 0
        self.system_bytes_counter = 0

        # Equipment state
        self.equipment_state = EquipmentState.EQUIPMENT_OFFLINE
        self.process_state = ProcessState.INIT
        self.control_state = "EQUIPMENT_OFFLINE"

        # Data storage
        self.equipment_constants: Dict[str, EquipmentConstant] = {}
        self.status_variables: Dict[str, DataVariable] = {}
        self.alarms: Dict[str, AlarmData] = {}
        self.event_callbacks: Dict[int, Callable] = {}

        # Connection
        self.connected = False
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

        self._initialize_equipment_constants()
        self._initialize_status_variables()

    def _initialize_equipment_constants(self):
        """Initialize equipment constants"""
        ecs = [
            EquipmentConstant("EC001", "ProcessTemp", 800.0, "°C", 200.0, 1200.0,
                            "Process temperature setpoint"),
            EquipmentConstant("EC002", "ProcessPressure", 10.0, "Torr", 0.1, 1000.0,
                            "Chamber pressure setpoint"),
            EquipmentConstant("EC003", "SiH4Flow", 100.0, "sccm", 0.0, 500.0,
                            "Silane flow rate"),
            EquipmentConstant("EC004", "N2Flow", 5000.0, "sccm", 0.0, 10000.0,
                            "Nitrogen carrier flow"),
            EquipmentConstant("EC005", "DepositionTime", 120.0, "sec", 10.0, 600.0,
                            "Film deposition duration"),
            EquipmentConstant("EC006", "RotationSpeed", 20.0, "rpm", 0.0, 100.0,
                            "Susceptor rotation speed"),
        ]
        self.equipment_constants = {ec.ec_id: ec for ec in ecs}

    def _initialize_status_variables(self):
        """Initialize status variables"""
        svs = [
            DataVariable("SV001", "ActualTemp", 0.0, "°C", datetime.utcnow()),
            DataVariable("SV002", "ActualPressure", 0.0, "Torr", datetime.utcnow()),
            DataVariable("SV003", "ActualSiH4Flow", 0.0, "sccm", datetime.utcnow()),
            DataVariable("SV004", "ActualN2Flow", 0.0, "sccm", datetime.utcnow()),
            DataVariable("SV005", "DepositionRate", 0.0, "nm/s", datetime.utcnow()),
            DataVariable("SV006", "WaferPosition", "LOAD_LOCK", "", datetime.utcnow()),
            DataVariable("SV007", "RecipeName", "", "", datetime.utcnow()),
            DataVariable("SV008", "LotID", "", "", datetime.utcnow()),
            DataVariable("SV009", "WaferID", "", "", datetime.utcnow()),
            DataVariable("SV010", "ProcessState", "IDLE", "", datetime.utcnow()),
        ]
        self.status_variables = {sv.var_id: sv for sv in svs}

    async def connect(self) -> bool:
        """Establish HSMS connection to equipment"""
        try:
            logger.info(f"Connecting to equipment at {self.host_ip}:{self.port}")
            self.reader, self.writer = await asyncio.open_connection(
                self.host_ip, self.port
            )
            self.connected = True
            self.equipment_state = EquipmentState.ATTEMPT_ONLINE
            logger.info(f"Connected to equipment {self.equipment_id}")

            # Send S1F1 (Are You There)
            await self.send_s1f1()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to equipment: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> bool:
        """Close connection"""
        try:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
            self.connected = False
            self.equipment_state = EquipmentState.EQUIPMENT_OFFLINE
            logger.info(f"Disconnected from equipment {self.equipment_id}")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False

    def _get_next_system_bytes(self) -> int:
        """Generate unique system bytes for message tracking"""
        self.system_bytes_counter = (self.system_bytes_counter + 1) % 0xFFFFFFFF
        return self.system_bytes_counter

    async def send_message(self, message: SECSMessage) -> bool:
        """Send SECS-II message to equipment"""
        try:
            if not self.connected:
                logger.error("Not connected to equipment")
                return False

            # Serialize message (simplified - real implementation needs proper SECS-II encoding)
            message_bytes = self._serialize_message(message)

            self.writer.write(message_bytes)
            await self.writer.drain()

            logger.debug(f"Sent S{message.stream}F{message.function}: {message.data}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _serialize_message(self, message: SECSMessage) -> bytes:
        """
        Serialize SECS-II message to bytes.
        Simplified implementation - production would use proper SECS-II encoding.
        """
        # Message header (10 bytes for HSMS)
        # [Length (4)] [Header (10)] [Data (variable)]

        data_json = json.dumps(message.data).encode('utf-8')
        data_length = len(data_json)

        # Simplified header
        header = bytearray(10)
        header[0] = self.device_id
        header[1] = message.stream
        header[2] = message.function | (0x80 if message.wait_bit else 0x00)
        header[4:8] = message.system_bytes.to_bytes(4, 'big')

        # Length field
        length_field = (data_length + 10).to_bytes(4, 'big')

        return length_field + header + data_json

    async def receive_message(self) -> Optional[SECSMessage]:
        """Receive SECS-II message from equipment"""
        try:
            if not self.connected or not self.reader:
                return None

            # Read length field (4 bytes)
            length_bytes = await self.reader.read(4)
            if not length_bytes:
                return None

            total_length = int.from_bytes(length_bytes, 'big')

            # Read header (10 bytes)
            header = await self.reader.read(10)
            device_id = header[0]
            stream = header[1] & 0x7F
            function = header[2] & 0x7F
            wait_bit = bool(header[2] & 0x80)
            system_bytes = int.from_bytes(header[4:8], 'big')

            # Read data
            data_length = total_length - 10
            data_bytes = await self.reader.read(data_length) if data_length > 0 else b''
            data = json.loads(data_bytes) if data_bytes else None

            message = SECSMessage(
                stream=stream,
                function=function,
                wait_bit=wait_bit,
                device_id=device_id,
                system_bytes=system_bytes,
                data=data,
                timestamp=datetime.utcnow()
            )

            logger.debug(f"Received S{stream}F{function}: {data}")

            # Process message
            await self._process_received_message(message)

            return message

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None

    async def _process_received_message(self, message: SECSMessage):
        """Process received SECS-II message"""
        msg_type = (message.stream, message.function)

        if msg_type == (1, 2):  # S1F2 - On Line Data
            await self._handle_s1f2(message)
        elif msg_type == (2, 16):  # S2F16 - EC Acknowledge
            await self._handle_s2f16(message)
        elif msg_type == (2, 42):  # S2F42 - Command Acknowledge
            await self._handle_s2f42(message)
        elif msg_type == (5, 1):  # S5F1 - Alarm Report
            await self._handle_s5f1(message)
        elif msg_type == (6, 11):  # S6F11 - Event Report
            await self._handle_s6f11(message)
        # Add more message handlers as needed

    # SECS-II Message Senders

    async def send_s1f1(self) -> bool:
        """S1F1 - Are You There Request"""
        message = SECSMessage(
            stream=1,
            function=1,
            wait_bit=True,
            device_id=self.device_id,
            system_bytes=self._get_next_system_bytes(),
            data=None,
            timestamp=datetime.utcnow()
        )
        return await self.send_message(message)

    async def send_s1f3(self, sv_ids: List[str]) -> bool:
        """S1F3 - Selected Equipment Status Request"""
        message = SECSMessage(
            stream=1,
            function=3,
            wait_bit=True,
            device_id=self.device_id,
            system_bytes=self._get_next_system_bytes(),
            data={"sv_ids": sv_ids},
            timestamp=datetime.utcnow()
        )
        return await self.send_message(message)

    async def send_s2f15(self, ec_updates: Dict[str, Any]) -> bool:
        """S2F15 - New Equipment Constant Send"""
        message = SECSMessage(
            stream=2,
            function=15,
            wait_bit=True,
            device_id=self.device_id,
            system_bytes=self._get_next_system_bytes(),
            data={"equipment_constants": ec_updates},
            timestamp=datetime.utcnow()
        )
        return await self.send_message(message)

    async def send_s2f41(self, command: str, parameters: Dict[str, Any]) -> bool:
        """S2F41 - Host Command Send"""
        message = SECSMessage(
            stream=2,
            function=41,
            wait_bit=True,
            device_id=self.device_id,
            system_bytes=self._get_next_system_bytes(),
            data={
                "command": command,
                "parameters": parameters
            },
            timestamp=datetime.utcnow()
        )
        return await self.send_message(message)

    async def send_s7f3(self, recipe_name: str, recipe_body: Dict[str, Any]) -> bool:
        """S7F3 - Process Program Send"""
        message = SECSMessage(
            stream=7,
            function=3,
            wait_bit=True,
            device_id=self.device_id,
            system_bytes=self._get_next_system_bytes(),
            data={
                "recipe_name": recipe_name,
                "recipe_body": recipe_body
            },
            timestamp=datetime.utcnow()
        )
        return await self.send_message(message)

    # SECS-II Message Handlers

    async def _handle_s1f2(self, message: SECSMessage):
        """Handle S1F2 - On Line Data"""
        logger.info(f"Equipment online: {message.data}")
        self.equipment_state = EquipmentState.ONLINE_REMOTE

    async def _handle_s2f16(self, message: SECSMessage):
        """Handle S2F16 - Equipment Constant Acknowledge"""
        ack_code = message.data.get("ack_code", 0)
        if ack_code == 0:
            logger.info("Equipment constants updated successfully")
        else:
            logger.warning(f"Equipment constant update failed: {ack_code}")

    async def _handle_s2f42(self, message: SECSMessage):
        """Handle S2F42 - Command Acknowledge"""
        command_result = message.data.get("result", {})
        logger.info(f"Command executed: {command_result}")

    async def _handle_s5f1(self, message: SECSMessage):
        """Handle S5F1 - Alarm Report Send"""
        alarm_data = message.data
        alarm = AlarmData(
            alarm_id=alarm_data.get("alarm_id", ""),
            alarm_code=alarm_data.get("alarm_code", 0),
            alarm_text=alarm_data.get("alarm_text", ""),
            severity=AlarmSeverity(alarm_data.get("severity", 1)),
            timestamp=datetime.utcnow(),
            equipment_id=self.equipment_id,
            state_change=alarm_data.get("state_change", False)
        )
        self.alarms[alarm.alarm_id] = alarm
        logger.warning(f"Alarm received: {alarm.alarm_text} (Code: {alarm.alarm_code})")

        # Send S5F2 acknowledge
        ack_message = SECSMessage(
            stream=5,
            function=2,
            wait_bit=False,
            device_id=self.device_id,
            system_bytes=message.system_bytes,
            data={"ack_code": 0},
            timestamp=datetime.utcnow()
        )
        await self.send_message(ack_message)

    async def _handle_s6f11(self, message: SECSMessage):
        """Handle S6F11 - Event Report Send"""
        event_data = message.data
        event_id = event_data.get("event_id", 0)

        logger.info(f"Event {event_id} reported: {event_data}")

        # Trigger callback if registered
        if event_id in self.event_callbacks:
            await self.event_callbacks[event_id](event_data)

        # Send S6F12 acknowledge
        ack_message = SECSMessage(
            stream=6,
            function=12,
            wait_bit=False,
            device_id=self.device_id,
            system_bytes=message.system_bytes,
            data={"ack_code": 0},
            timestamp=datetime.utcnow()
        )
        await self.send_message(ack_message)

    # High-level Equipment Control

    async def start_process(self, recipe_name: str, lot_id: str, wafer_id: str) -> bool:
        """Start CVD process with specified recipe"""
        try:
            logger.info(f"Starting process: Recipe={recipe_name}, Lot={lot_id}, Wafer={wafer_id}")

            # Update status variables
            self.status_variables["SV007"].value = recipe_name
            self.status_variables["SV008"].value = lot_id
            self.status_variables["SV009"].value = wafer_id
            self.status_variables["SV010"].value = "EXECUTING"
            self.process_state = ProcessState.EXECUTING

            # Send START command via S2F41
            result = await self.send_s2f41("START", {
                "recipe": recipe_name,
                "lot_id": lot_id,
                "wafer_id": wafer_id
            })

            return result

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            return False

    async def stop_process(self) -> bool:
        """Stop current process"""
        try:
            logger.info("Stopping process")
            self.process_state = ProcessState.PAUSE
            result = await self.send_s2f41("STOP", {})
            return result
        except Exception as e:
            logger.error(f"Failed to stop process: {e}")
            return False

    async def update_recipe_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update equipment constants (recipe parameters)"""
        try:
            logger.info(f"Updating recipe parameters: {parameters}")

            # Update local EC values
            for ec_id, value in parameters.items():
                if ec_id in self.equipment_constants:
                    self.equipment_constants[ec_id].value = value

            # Send to equipment via S2F15
            result = await self.send_s2f15(parameters)
            return result

        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False

    async def get_status_variables(self, sv_ids: Optional[List[str]] = None) -> Dict[str, DataVariable]:
        """Get current status variable values"""
        if sv_ids is None:
            return self.status_variables
        else:
            return {sv_id: self.status_variables[sv_id] for sv_id in sv_ids if sv_id in self.status_variables}

    async def get_active_alarms(self) -> List[AlarmData]:
        """Get list of active alarms"""
        return [alarm for alarm in self.alarms.values() if alarm.state_change]

    def register_event_callback(self, event_id: int, callback: Callable):
        """Register callback for specific event"""
        self.event_callbacks[event_id] = callback
        logger.info(f"Registered callback for event {event_id}")

    async def message_loop(self):
        """Main message receiving loop"""
        logger.info("Starting SECS/GEM message loop")
        while self.connected:
            try:
                message = await self.receive_message()
                if message is None:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(1.0)
