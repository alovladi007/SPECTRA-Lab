# services/instruments/app/drivers/plugin_manager.py

“””
Plugin Architecture for Dynamic Driver Loading

Provides:

- Dynamic discovery and loading of instrument drivers
- Version compatibility checking
- Capability registration and discovery
- Driver lifecycle management
- Hot-reload support for development
  “””

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Type, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import yaml
import json
from enum import Enum
from packaging import version

# ============================================================================

# Plugin Metadata

# ============================================================================

@dataclass
class PluginMetadata:
“”“Metadata for an instrument driver plugin”””
name: str
version: str
author: str
description: str

# Compatibility
min_platform_version: str = "1.0.0"
max_platform_version: Optional[str] = None

# Capabilities
supported_methods: List[str] = field(default_factory=list)
supported_models: List[str] = field(default_factory=list)

# Driver class
driver_class: str = ""  # e.g., "keithley.Keithley2400Driver"

# Dependencies
dependencies: List[str] = field(default_factory=list)

# Configuration
config_schema: Optional[Dict[str, Any]] = None

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
    """Create from dictionary"""
    return cls(**data)

@classmethod
def from_yaml(cls, yaml_path: Path) -> 'PluginMetadata':
    """Load from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return cls.from_dict(data)

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary"""
    return {
        'name': self.name,
        'version': self.version,
        'author': self.author,
        'description': self.description,
        'min_platform_version': self.min_platform_version,
        'max_platform_version': self.max_platform_version,
        'supported_methods': self.supported_methods,
        'supported_models': self.supported_models,
        'driver_class': self.driver_class,
        'dependencies': self.dependencies,
        'config_schema': self.config_schema
    }

class PluginStatus(str, Enum):
“”“Plugin loading status”””
DISCOVERED = “discovered”
LOADED = “loaded”
ACTIVE = “active”
ERROR = “error”
DISABLED = “disabled”

@dataclass
class PluginInfo:
“”“Information about a loaded plugin”””
metadata: PluginMetadata
status: PluginStatus
driver_class: Optional[Type] = None
error_message: Optional[str] = None
load_time: Optional[float] = None

# ============================================================================

# Plugin Interface

# ============================================================================

class InstrumentDriver(ABC):
“””
Base interface that all instrument drivers must implement

This defines the contract that the platform expects from drivers.
"""

@abstractmethod
def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Initialize driver
    
    Args:
        resource_name: VISA resource string or connection identifier
        config: Optional configuration dictionary
    """
    pass

@abstractmethod
def connect(self) -> bool:
    """
    Establish connection to instrument
    
    Returns:
        True if connection successful
    """
    pass

@abstractmethod
def disconnect(self) -> bool:
    """
    Close connection to instrument
    
    Returns:
        True if disconnection successful
    """
    pass

@abstractmethod
def reset(self) -> None:
    """Reset instrument to default state"""
    pass

@abstractmethod
def get_identity(self) -> Dict[str, str]:
    """
    Get instrument identity
    
    Returns:
        Dictionary with manufacturer, model, serial, firmware keys
    """
    pass

@abstractmethod
def get_capabilities(self) -> List[str]:
    """
    Get list of supported methods
    
    Returns:
        List of method names (e.g., ['iv_sweep', 'cv_measurement'])
    """
    pass

@abstractmethod
def configure(self, method: str, params: Dict[str, Any]) -> None:
    """
    Configure instrument for a specific method
    
    Args:
        method: Method name
        params: Method-specific parameters
    """
    pass

@abstractmethod
def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform measurement
    
    Args:
        method: Method name
        params: Measurement parameters
        
    Returns:
        Dictionary with measurement results
    """
    pass

@abstractmethod
def abort(self) -> None:
    """Abort ongoing measurement"""
    pass

@abstractmethod
def get_status(self) -> Dict[str, Any]:
    """
    Get instrument status
    
    Returns:
        Dictionary with status information
    """
    pass

def __enter__(self):
    """Context manager entry"""
    self.connect()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit"""
    self.disconnect()

# ============================================================================

# Plugin Manager

# ============================================================================

class PluginManager:
“””
Manages instrument driver plugins

Features:
- Discovers plugins from plugin directories
- Loads and validates plugins
- Version compatibility checking
- Capability registry
- Hot-reload for development
"""

def __init__(
    self,
    plugin_dirs: Optional[List[Path]] = None,
    platform_version: str = "1.0.0",
    logger: Optional[logging.Logger] = None
):
    """
    Initialize plugin manager
    
    Args:
        plugin_dirs: List of directories to search for plugins
        platform_version: Current platform version
        logger: Logger instance
    """
    self.platform_version = platform_version
    self.logger = logger or logging.getLogger(__name__)
    
    # Default plugin directories
    if plugin_dirs is None:
        base_dir = Path(__file__).parent
        plugin_dirs = [
            base_dir / "builtin",  # Built-in drivers
            base_dir / "plugins",  # User-added drivers
        ]
    
    self.plugin_dirs = [Path(d) for d in plugin_dirs]
    
    # Plugin registry
    self.plugins: Dict[str, PluginInfo] = {}
    
    # Capability index (method -> list of driver names)
    self.capability_index: Dict[str, List[str]] = {}
    
    # Model index (model -> driver name)
    self.model_index: Dict[str, str] = {}

def discover_plugins(self) -> List[str]:
    """
    Discover all plugins in plugin directories
    
    Returns:
        List of discovered plugin names
    """
    discovered = []
    
    for plugin_dir in self.plugin_dirs:
        if not plugin_dir.exists():
            self.logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            continue
        
        # Look for plugin.yaml files
        for yaml_file in plugin_dir.rglob("plugin.yaml"):
            try:
                metadata = PluginMetadata.from_yaml(yaml_file)
                
                # Check if already loaded
                if metadata.name in self.plugins:
                    self.logger.warning(f"Plugin '{metadata.name}' already discovered")
                    continue
                
                # Add to registry
                self.plugins[metadata.name] = PluginInfo(
                    metadata=metadata,
                    status=PluginStatus.DISCOVERED
                )
                
                discovered.append(metadata.name)
                self.logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")
                
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {yaml_file}: {e}")
    
    return discovered

def load_plugin(self, plugin_name: str) -> bool:
    """
    Load a specific plugin
    
    Args:
        plugin_name: Name of plugin to load
        
    Returns:
        True if loaded successfully
    """
    if plugin_name not in self.plugins:
        self.logger.error(f"Plugin '{plugin_name}' not found")
        return False
    
    plugin_info = self.plugins[plugin_name]
    metadata = plugin_info.metadata
    
    try:
        # Check version compatibility
        if not self._check_compatibility(metadata):
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = "Version incompatible"
            return False
        
        # Load driver class
        module_name, class_name = metadata.driver_class.rsplit('.', 1)
        
        # Import module
        # Try to find the module in plugin directories
        module_path = None
        for plugin_dir in self.plugin_dirs:
            potential_path = plugin_dir / f"{module_name.replace('.', '/')}.py"
            if potential_path.exists():
                module_path = potential_path
                break
        
        if module_path is None:
            raise ImportError(f"Cannot find module {module_name}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {module_name}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get driver class
        driver_class = getattr(module, class_name)
        
        # Verify it implements InstrumentDriver interface
        if not issubclass(driver_class, InstrumentDriver):
            raise TypeError(f"{class_name} does not implement InstrumentDriver interface")
        
        # Update plugin info
        plugin_info.driver_class = driver_class
        plugin_info.status = PluginStatus.LOADED
        
        # Update capability index
        for method in metadata.supported_methods:
            if method not in self.capability_index:
                self.capability_index[method] = []
            self.capability_index[method].append(plugin_name)
        
        # Update model index
        for model in metadata.supported_models:
            self.model_index[model] = plugin_name
        
        self.logger.info(f"Loaded plugin: {plugin_name}")
        return True
        
    except Exception as e:
        plugin_info.status = PluginStatus.ERROR
        plugin_info.error_message = str(e)
        self.logger.error(f"Failed to load plugin '{plugin_name}': {e}")
        return False

def load_all_plugins(self) -> Dict[str, bool]:
    """
    Load all discovered plugins
    
    Returns:
        Dictionary of plugin_name -> success status
    """
    results = {}
    for plugin_name in self.plugins.keys():
        if self.plugins[plugin_name].status == PluginStatus.DISCOVERED:
            results[plugin_name] = self.load_plugin(plugin_name)
    return results

def get_driver(self, plugin_name: str, resource_name: str, config: Optional[Dict[str, Any]] = None) -> InstrumentDriver:
    """
    Create driver instance
    
    Args:
        plugin_name: Name of plugin
        resource_name: VISA resource string
        config: Optional configuration
        
    Returns:
        Driver instance
    """
    if plugin_name not in self.plugins:
        raise ValueError(f"Plugin '{plugin_name}' not found")
    
    plugin_info = self.plugins[plugin_name]
    
    if plugin_info.status != PluginStatus.LOADED:
        raise RuntimeError(f"Plugin '{plugin_name}' not loaded (status: {plugin_info.status})")
    
    if plugin_info.driver_class is None:
        raise RuntimeError(f"Plugin '{plugin_name}' has no driver class")
    
    # Create instance
    return plugin_info.driver_class(resource_name, config)

def find_driver_for_model(self, model: str) -> Optional[str]:
    """
    Find driver plugin for a specific instrument model
    
    Args:
        model: Instrument model string
        
    Returns:
        Plugin name or None if not found
    """
    return self.model_index.get(model)

def find_drivers_for_method(self, method: str) -> List[str]:
    """
    Find all drivers that support a specific method
    
    Args:
        method: Method name
        
    Returns:
        List of plugin names
    """
    return self.capability_index.get(method, [])

def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
    """Get information about a plugin"""
    return self.plugins.get(plugin_name)

def list_plugins(self) -> Dict[str, PluginInfo]:
    """List all plugins"""
    return self.plugins.copy()

def _check_compatibility(self, metadata: PluginMetadata) -> bool:
    """Check if plugin is compatible with current platform version"""
    try:
        platform_ver = version.parse(self.platform_version)
        min_ver = version.parse(metadata.min_platform_version)
        
        if platform_ver < min_ver:
            self.logger.warning(
                f"Plugin requires platform version >= {metadata.min_platform_version}, "
                f"current version is {self.platform_version}"
            )
            return False
        
        if metadata.max_platform_version:
            max_ver = version.parse(metadata.max_platform_version)
            if platform_ver > max_ver:
                self.logger.warning(
                    f"Plugin requires platform version <= {metadata.max_platform_version}, "
                    f"current version is {self.platform_version}"
                )
                return False
        
        return True
        
    except Exception as e:
        self.logger.error(f"Version compatibility check failed: {e}")
        return False

# ============================================================================

# Plugin Decorators

# ============================================================================

def driver_plugin(
name: str,
version: str,
author: str,
description: str,
supported_methods: List[str],
supported_models: List[str],
**kwargs
):
“””
Decorator to mark a class as a driver plugin

Example:
    @driver_plugin(
        name="keithley_2400",
        version="1.0.0",
        author="Lab Team",
        description="Keithley 2400 SMU Driver",
        supported_methods=["iv_sweep", "cv_measurement"],
        supported_models=["2400", "2401"]
    )
    class Keithley2400Driver(InstrumentDriver):
        ...
"""
def decorator(cls):
    # Attach metadata to class
    cls._plugin_metadata = PluginMetadata(
        name=name,
        version=version,
        author=author,
        description=description,
        supported_methods=supported_methods,
        supported_models=supported_models,
        driver_class=f"{cls.__module__}.{cls.__name__}",
        **kwargs
    )
    return cls

return decorator

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate plugin architecture”””
print(”=” * 80)
print(“Plugin Architecture - Example Usage”)
print(”=” * 80)

# 1. Create plugin manager
print("\n1. Creating Plugin Manager:")
manager = PluginManager(platform_version="1.0.0")
print(f"   Platform version: {manager.platform_version}")
print(f"   Plugin directories: {[str(d) for d in manager.plugin_dirs]}")

# 2. Example plugin metadata
print("\n2. Plugin Metadata Structure:")
metadata = PluginMetadata(
    name="keithley_2400",
    version="1.0.0",
    author="Lab Team",
    description="Keithley 2400 SMU Driver",
    min_platform_version="1.0.0",
    supported_methods=["iv_sweep", "cv_measurement"],
    supported_models=["2400", "2401", "2410"],
    driver_class="keithley.Keithley2400Driver"
)
print(f"   Name: {metadata.name}")
print(f"   Version: {metadata.version}")
print(f"   Methods: {metadata.supported_methods}")
print(f"   Models: {metadata.supported_models}")

# 3. Plugin lifecycle
print("\n3. Plugin Lifecycle:")
print(f"   States: {[s.value for s in PluginStatus]}")
print("   Flow: discovered → loaded → active")

# 4. Capability discovery
print("\n4. Capability Discovery:")
print("   Example: Find drivers supporting 'iv_sweep'")
print("   manager.find_drivers_for_method('iv_sweep')")
print("   → ['keithley_2400', 'keysight_b2900', ...]")

# 5. Model matching
print("\n5. Model Matching:")
print("   Example: Find driver for 'Keithley 2400'")
print("   manager.find_driver_for_model('2400')")
print("   → 'keithley_2400'")

print("\n" + "=" * 80)
print("Plugin Architecture demonstration complete!")
print("=" * 80)
print("\nTo add a new driver:")
print("1. Create driver class inheriting from InstrumentDriver")
print("2. Create plugin.yaml with metadata")
print("3. Place in plugins/ directory")
print("4. Manager will auto-discover on startup")

if **name** == “**main**”:
# Configure logging
logging.basicConfig(level=logging.INFO)

# Run example
example_usage()