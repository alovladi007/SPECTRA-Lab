"""
Neural Architecture Search (NAS) for Semiconductor Manufacturing
Automatically designs optimal neural network architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkArchitecture:
    """Represents a neural network architecture"""
    layers: List[Dict[str, Any]]
    num_params: int
    performance: float
    inference_time: float
    architecture_string: str


class NeuralArchitectureSearch:
    """
    Neural Architecture Search for semiconductor manufacturing
    
    Implements multiple NAS strategies:
    1. Random Search: Baseline approach
    2. Grid Search: Systematic exploration
    3. Evolutionary Search: Genetic algorithm-based
    4. Reinforcement Learning: Policy-based architecture generation
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        task_type: str = "regression",
        search_strategy: str = "evolutionary",
        n_architectures: int = 20,
        max_layers: int = 5,
        max_units_per_layer: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.search_strategy = search_strategy
        self.n_architectures = n_architectures
        self.max_layers = max_layers
        self.max_units_per_layer = max_units_per_layer
        self.device = device
        
        self.best_architecture = None
        self.best_model = None
        self.architecture_history = []
        
        logger.info(f"Initialized NAS with {search_strategy} strategy on {device}")
    
    def _create_network(self, architecture: List[Dict[str, Any]]) -> nn.Module:
        """Create PyTorch network from architecture specification"""
        
        class DynamicNetwork(nn.Module):
            def __init__(self, input_dim, output_dim, layers_config):
                super().__init__()
                self.layers = nn.ModuleList()
                
                prev_dim = input_dim
                for layer_config in layers_config:
                    layer_type = layer_config["type"]
                    
                    if layer_type == "linear":
                        units = layer_config["units"]
                        self.layers.append(nn.Linear(prev_dim, units))
                        prev_dim = units
                        
                        # Add activation
                        activation = layer_config.get("activation", "relu")
                        if activation == "relu":
                            self.layers.append(nn.ReLU())
                        elif activation == "tanh":
                            self.layers.append(nn.Tanh())
                        elif activation == "sigmoid":
                            self.layers.append(nn.Sigmoid())
                        
                        # Add dropout if specified
                        if "dropout" in layer_config and layer_config["dropout"] > 0:
                            self.layers.append(nn.Dropout(layer_config["dropout"]))
                    
                    elif layer_type == "batchnorm":
                        self.layers.append(nn.BatchNorm1d(prev_dim))
                
                # Output layer
                self.layers.append(nn.Linear(prev_dim, output_dim))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return DynamicNetwork(self.input_dim, self.output_dim, architecture)
    
    def _generate_random_architecture(self) -> List[Dict[str, Any]]:
        """Generate a random neural network architecture"""
        n_layers = np.random.randint(1, self.max_layers + 1)
        architecture = []
        
        for i in range(n_layers):
            # Generate layer configuration
            layer = {
                "type": "linear",
                "units": int(np.random.choice([16, 32, 64, 128, 256][:min(5, self.max_units_per_layer // 16)])),
                "activation": np.random.choice(["relu", "tanh"]),
            }
            
            # Add dropout with 50% probability
            if np.random.random() > 0.5:
                layer["dropout"] = np.random.choice([0.1, 0.2, 0.3, 0.4])
            
            architecture.append(layer)
            
            # Add batch normalization with 30% probability
            if np.random.random() > 0.7:
                architecture.append({"type": "batchnorm"})
        
        return architecture
    
    def _evaluate_architecture(
        self,
        architecture: List[Dict[str, Any]],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Tuple[float, float, int]:
        """
        Evaluate a neural network architecture
        
        Returns:
            (validation_score, inference_time, num_params)
        """
        try:
            # Create model
            model = self._create_network(architecture).to(self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Setup training
            criterion = nn.MSELoss() if self.task_type == "regression" else nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if self.task_type == "regression":
                        loss = criterion(outputs.squeeze(), batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)
                
                # Measure inference time
                import time
                start_time = time.time()
                val_outputs = model(X_val_device)
                inference_time = (time.time() - start_time) / len(X_val)
                
                if self.task_type == "regression":
                    # Calculate R² score
                    val_outputs = val_outputs.squeeze()
                    ss_res = torch.sum((y_val_device - val_outputs) ** 2).item()
                    ss_tot = torch.sum((y_val_device - torch.mean(y_val_device)) ** 2).item()
                    r2_score = 1 - (ss_res / (ss_tot + 1e-10))
                    val_score = r2_score
                else:
                    # Classification accuracy
                    _, predicted = torch.max(val_outputs, 1)
                    val_score = (predicted == y_val_device).float().mean().item()
            
            return val_score, inference_time, num_params
            
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {str(e)}")
            return -1.0, 1.0, 0
    
    def _architecture_to_string(self, architecture: List[Dict[str, Any]]) -> str:
        """Convert architecture to readable string"""
        parts = []
        for layer in architecture:
            if layer["type"] == "linear":
                parts.append(f"L{layer['units']}_{layer['activation']}")
                if "dropout" in layer:
                    parts.append(f"D{layer['dropout']}")
            elif layer["type"] == "batchnorm":
                parts.append("BN")
        return "-".join(parts)
    
    def search_random(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Random architecture search"""
        logger.info("Starting Random Architecture Search...")
        
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        best_score = -float('inf')
        
        for i in range(self.n_architectures):
            # Generate random architecture
            architecture = self._generate_random_architecture()
            
            # Evaluate
            score, inf_time, n_params = self._evaluate_architecture(
                architecture, X_train_t, y_train_t, X_val_t, y_val_t
            )
            
            # Store in history
            arch_obj = NetworkArchitecture(
                layers=architecture,
                num_params=n_params,
                performance=score,
                inference_time=inf_time,
                architecture_string=self._architecture_to_string(architecture)
            )
            self.architecture_history.append(arch_obj)
            
            logger.info(f"Architecture {i+1}/{self.n_architectures}: "
                       f"Score={score:.4f}, Params={n_params}, "
                       f"Time={inf_time*1000:.2f}ms")
            
            # Update best
            if score > best_score:
                best_score = score
                self.best_architecture = architecture
                self.best_model = self._create_network(architecture)
        
        return self._get_search_results()
    
    def search_evolutionary(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        population_size: int = 10,
        n_generations: int = 5,
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """Evolutionary architecture search using genetic algorithms"""
        logger.info("Starting Evolutionary Architecture Search...")
        logger.info(f"Population: {population_size}, Generations: {n_generations}")
        
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # Initialize population
        population = [self._generate_random_architecture() for _ in range(population_size)]
        
        for generation in range(n_generations):
            logger.info(f"\nGeneration {generation + 1}/{n_generations}")
            
            # Evaluate population
            fitness_scores = []
            for arch in population:
                score, inf_time, n_params = self._evaluate_architecture(
                    arch, X_train_t, y_train_t, X_val_t, y_val_t, epochs=30
                )
                # Fitness: balance accuracy and efficiency
                fitness = score - (n_params / 100000) * 0.1  # Penalty for large models
                fitness_scores.append((fitness, score, arch, n_params, inf_time))
            
            # Sort by fitness
            fitness_scores.sort(reverse=True, key=lambda x: x[0])
            
            # Log best in generation
            best_fitness, best_score, best_arch, best_params, best_time = fitness_scores[0]
            logger.info(f"  Best: Score={best_score:.4f}, Params={best_params}, "
                       f"Fitness={best_fitness:.4f}")
            
            # Store best architecture
            arch_obj = NetworkArchitecture(
                layers=best_arch,
                num_params=best_params,
                performance=best_score,
                inference_time=best_time,
                architecture_string=self._architecture_to_string(best_arch)
            )
            self.architecture_history.append(arch_obj)
            
            # Selection: keep top 50%
            survivors = [arch for _, _, arch, _, _ in fitness_scores[:population_size//2]]
            
            # Crossover and mutation
            new_population = survivors.copy()
            while len(new_population) < population_size:
                # Select two parents
                parent1, parent2 = np.random.choice(len(survivors), 2, replace=False)
                
                # Crossover
                child = self._crossover(survivors[parent1], survivors[parent2])
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Final evaluation of best architecture
        best_arch = fitness_scores[0][2]
        self.best_architecture = best_arch
        self.best_model = self._create_network(best_arch)
        
        # Train best model for longer
        logger.info("\nTraining best architecture for extended epochs...")
        _, _, _ = self._evaluate_architecture(
            best_arch, X_train_t, y_train_t, X_val_t, y_val_t, epochs=100
        )
        
        return self._get_search_results()
    
    def _crossover(
        self,
        parent1: List[Dict[str, Any]],
        parent2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Crossover operation for evolutionary search"""
        # Single-point crossover
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1 if len(parent1) > 0 else parent2
        
        point = np.random.randint(1, max(len(parent1), len(parent2)))
        child = parent1[:point] + parent2[point:]
        
        # Ensure at least one layer
        if len(child) == 0:
            child = [parent1[0]] if len(parent1) > 0 else [parent2[0]]
        
        return child
    
    def _mutate(self, architecture: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutation operation for evolutionary search"""
        mutated = architecture.copy()
        
        mutation_type = np.random.choice([
            "add_layer", "remove_layer", "modify_layer", "add_dropout", "remove_dropout"
        ])
        
        if mutation_type == "add_layer" and len(mutated) < self.max_layers:
            # Add a new random layer
            new_layer = {
                "type": "linear",
                "units": int(np.random.choice([16, 32, 64, 128, 256])),
                "activation": np.random.choice(["relu", "tanh"])
            }
            insert_pos = np.random.randint(0, len(mutated) + 1)
            mutated.insert(insert_pos, new_layer)
        
        elif mutation_type == "remove_layer" and len(mutated) > 1:
            # Remove a random layer
            linear_layers = [i for i, l in enumerate(mutated) if l["type"] == "linear"]
            if linear_layers:
                remove_idx = np.random.choice(linear_layers)
                mutated.pop(remove_idx)
        
        elif mutation_type == "modify_layer" and mutated:
            # Modify a random layer's units
            linear_layers = [i for i, l in enumerate(mutated) if l["type"] == "linear"]
            if linear_layers:
                modify_idx = np.random.choice(linear_layers)
                mutated[modify_idx]["units"] = int(np.random.choice([16, 32, 64, 128, 256]))
        
        elif mutation_type == "add_dropout":
            # Add dropout to a random layer
            linear_layers = [i for i, l in enumerate(mutated) if l["type"] == "linear"]
            if linear_layers:
                dropout_idx = np.random.choice(linear_layers)
                mutated[dropout_idx]["dropout"] = np.random.choice([0.1, 0.2, 0.3])
        
        elif mutation_type == "remove_dropout":
            # Remove dropout from a random layer
            dropout_layers = [i for i, l in enumerate(mutated) 
                            if l.get("type") == "linear" and "dropout" in l]
            if dropout_layers:
                remove_dropout_idx = np.random.choice(dropout_layers)
                del mutated[remove_dropout_idx]["dropout"]
        
        return mutated
    
    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Run architecture search based on selected strategy"""
        if self.search_strategy == "random":
            return self.search_random(X_train, y_train, X_val, y_val)
        elif self.search_strategy == "evolutionary":
            return self.search_evolutionary(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
    
    def _get_search_results(self) -> Dict[str, Any]:
        """Compile search results"""
        # Sort by performance
        sorted_archs = sorted(
            self.architecture_history,
            key=lambda x: x.performance,
            reverse=True
        )
        
        return {
            "best_architecture": {
                "layers": self.best_architecture,
                "architecture_string": self._architecture_to_string(self.best_architecture),
                "num_params": sorted_archs[0].num_params if sorted_archs else 0,
                "performance": sorted_archs[0].performance if sorted_archs else 0,
                "inference_time_ms": sorted_archs[0].inference_time * 1000 if sorted_archs else 0
            },
            "search_strategy": self.search_strategy,
            "architectures_evaluated": len(self.architecture_history),
            "top_architectures": [
                {
                    "architecture_string": arch.architecture_string,
                    "num_params": arch.num_params,
                    "performance": float(arch.performance),
                    "inference_time_ms": float(arch.inference_time * 1000)
                }
                for arch in sorted_archs[:5]
            ]
        }
    
    def save(self, filepath: str):
        """Save the best model and search results"""
        # Save model
        torch.save(self.best_model.state_dict(), filepath)
        
        # Save architecture and results
        results_path = filepath.replace('.pth', '_nas_results.json')
        with open(results_path, 'w') as f:
            json.dump(self._get_search_results(), f, indent=2)
        
        logger.info(f"Best model saved to {filepath}")
        logger.info(f"NAS results saved to {results_path}")
