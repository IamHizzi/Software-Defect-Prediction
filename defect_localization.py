"""
Defect Localization Module
Phase 2: GAT-based defect localization on augmented ASTs
"""

import ast
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using simplified localization.")


class ASTParser:
    """Parse Python code into augmented AST graph"""
    
    def __init__(self):
        self.node_types = {}
        self.type_counter = 0
        
    def get_node_type_id(self, node_type):
        """Get unique ID for node type"""
        if node_type not in self.node_types:
            self.node_types[node_type] = self.type_counter
            self.type_counter += 1
        return self.node_types[node_type]
    
    def extract_node_features(self, node, defect_prob=0.0, metrics=None):
        """Extract features for an AST node"""
        features = []
        
        # Node type (one-hot would be better, but using ID for simplicity)
        node_type_id = self.get_node_type_id(type(node).__name__)
        features.append(node_type_id)
        
        # Defect probability from Phase 1
        features.append(defect_prob)
        
        # Line number
        features.append(getattr(node, 'lineno', 0))
        
        # Column offset
        features.append(getattr(node, 'col_offset', 0))
        
        # Node-specific features
        if isinstance(node, ast.FunctionDef):
            features.append(len(node.args.args))  # Number of arguments
            features.append(len(node.body))  # Body length
        else:
            features.extend([0, 0])
        
        # Add code metrics if provided
        if metrics:
            features.append(metrics.get('cyclomatic_complexity', 0))
            features.append(metrics.get('max_nesting_depth', 0))
        else:
            features.extend([0, 0])
        
        return features
    
    def build_graph_from_ast(self, code_text, defect_prob=0.0, metrics=None):
        """
        Build graph representation from AST
        Returns node features and edge indices
        """
        try:
            tree = ast.parse(code_text)
        except:
            # Return minimal graph if parsing fails
            if TORCH_AVAILABLE:
                return torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float), torch.tensor([[0], [0]], dtype=torch.long)
            else:
                return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32), np.array([[0], [0]], dtype=np.int64)
        
        # Build node list and edge list
        nodes = []
        edges = []
        node_map = {}
        node_id = 0
        
        def traverse(node, parent_id=None):
            nonlocal node_id
            
            current_id = node_id
            node_map[id(node)] = current_id
            
            # Extract features
            features = self.extract_node_features(node, defect_prob, metrics)
            nodes.append(features)
            
            # Add edge from parent
            if parent_id is not None:
                edges.append([parent_id, current_id])
            
            node_id += 1
            
            # Traverse children
            for child in ast.iter_child_nodes(node):
                traverse(child, current_id)
        
        traverse(tree)
        
        # Convert to tensors
        if TORCH_AVAILABLE:
            node_features = torch.tensor(nodes, dtype=torch.float)
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            node_features = np.array(nodes, dtype=np.float32)
            if len(edges) > 0:
                edge_index = np.array(edges, dtype=np.int64).T
            else:
                edge_index = np.array([[0], [0]], dtype=np.int64)
        
        return node_features, edge_index


if TORCH_AVAILABLE:
    class GATDefectLocalizer(nn.Module):
        """
        Graph Attention Network for defect localization
        """
        
        def __init__(self, in_channels, hidden_channels=64, num_heads=4, dropout=0.3):
            super(GATDefectLocalizer, self).__init__()
            
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
            self.classifier = nn.Linear(hidden_channels, 1)
            self.dropout = dropout
            
        def forward(self, x, edge_index):
            # First GAT layer
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Second GAT layer
            x = self.gat2(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Node-level classification
            x = self.classifier(x)
            x = torch.sigmoid(x)
            
            return x
else:
    # Dummy class when PyTorch not available
    class GATDefectLocalizer:
        def __init__(self, *args, **kwargs):
            pass


class DefectLocalizer:
    """
    Defect localization using GAT on augmented ASTs
    """
    
    def __init__(self, hidden_channels=64, num_heads=4):
        self.parser = ASTParser()
        self.model = None
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
    def prepare_graph(self, code_text, defect_prob=0.0, metrics=None):
        """Prepare graph data from code"""
        node_features, edge_index = self.parser.build_graph_from_ast(
            code_text, defect_prob, metrics
        )
        
        if TORCH_AVAILABLE:
            data = Data(x=node_features, edge_index=edge_index)
            return data
        else:
            # Return dict for non-torch mode
            return {
                'x': node_features,
                'edge_index': edge_index
            }
    
    def train_model(self, train_data_list, epochs=50, lr=0.01):
        """
        Train GAT model
        train_data_list: list of (Data, labels) tuples
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Skipping GAT training.")
            return
        
        if not train_data_list:
            print("No training data provided")
            return
        
        # Initialize model
        sample_data = train_data_list[0][0]
        in_channels = sample_data.x.size(1)
        self.model = GATDefectLocalizer(
            in_channels, 
            self.hidden_channels, 
            self.num_heads
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, labels in train_data_list:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = criterion(out.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_data_list)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def localize_defects(self, code_text, defect_prob=0.0, metrics=None, threshold=0.5):
        """
        Localize defects in code
        Returns node scores and top defective nodes
        """
        if not TORCH_AVAILABLE:
            # Fallback: use heuristic-based localization
            return self._heuristic_localize(code_text, defect_prob, metrics, threshold)
        
        if self.model is None:
            # Initialize with dummy model if not trained
            data = self.prepare_graph(code_text, defect_prob, metrics)
            in_channels = data.x.size(1)
            self.model = GATDefectLocalizer(
                in_channels, 
                self.hidden_channels, 
                self.num_heads
            ).to(self.device)
        
        data = self.prepare_graph(code_text, defect_prob, metrics)
        data = data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            scores = self.model(data.x, data.edge_index)
            scores = scores.cpu().numpy().flatten()
        
        # Find top defective nodes
        defective_nodes = np.where(scores > threshold)[0]
        top_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        
        results = {
            'node_scores': scores,
            'defective_nodes': defective_nodes,
            'top_nodes': top_scores
        }
        
        return results
    
    def _heuristic_localize(self, code_text, defect_prob, metrics, threshold):
        """Heuristic-based localization when PyTorch is not available"""
        try:
            tree = ast.parse(code_text)
            nodes = list(ast.walk(tree))
            num_nodes = len(nodes)
            
            # Simple heuristic: assign higher scores to complex nodes
            scores = np.ones(num_nodes) * defect_prob * 0.5
            
            for i, node in enumerate(nodes):
                # Increase score for potentially problematic patterns
                if isinstance(node, ast.Div):
                    scores[i] = min(1.0, defect_prob * 1.5)
                elif isinstance(node, (ast.Subscript, ast.Index)):
                    scores[i] = min(1.0, defect_prob * 1.3)
                elif isinstance(node, ast.While):
                    scores[i] = min(1.0, defect_prob * 1.2)
                elif isinstance(node, ast.ExceptHandler):
                    if not node.type:  # bare except
                        scores[i] = min(1.0, defect_prob * 1.4)
            
            defective_nodes = np.where(scores > threshold)[0]
            top_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'node_scores': scores,
                'defective_nodes': defective_nodes,
                'top_nodes': top_scores
            }
        except:
            # Return minimal result
            return {
                'node_scores': np.array([defect_prob]),
                'defective_nodes': np.array([0]),
                'top_nodes': [(0, defect_prob)]
            }
    
    def get_defective_lines(self, code_text, defect_prob=0.0, metrics=None):
        """Get line numbers of potentially defective code"""
        try:
            tree = ast.parse(code_text)
            results = self.localize_defects(code_text, defect_prob, metrics)
            
            # Map node indices to line numbers
            nodes_list = list(ast.walk(tree))
            defective_lines = []
            
            for node_idx in results['defective_nodes']:
                if node_idx < len(nodes_list):
                    node = nodes_list[node_idx]
                    if hasattr(node, 'lineno'):
                        defective_lines.append(node.lineno)
            
            return sorted(set(defective_lines))
        except:
            return []


def generate_synthetic_ast_data(n_samples=100):
    """Generate synthetic training data for GAT"""
    training_data = []
    
    code_samples = [
        # Defective code samples
        """
def buggy_function(x):
    if x > 0:
        return x / 0  # Bug: division by zero
    return x
""",
        """
def another_bug(items):
    total = 0
    for i in range(len(items)):
        total += items[i+1]  # Bug: index out of bounds
    return total
""",
        # Clean code samples
        """
def clean_function(x):
    if x > 0:
        return x * 2
    return x
""",
        """
def another_clean(items):
    return sum(items)
"""
    ]
    
    parser = ASTParser()
    
    for code in code_samples[:n_samples]:
        # Parse to graph
        node_features, edge_index = parser.build_graph_from_ast(code)
        data = Data(x=node_features, edge_index=edge_index)
        
        # Create synthetic labels (some nodes are buggy)
        num_nodes = node_features.size(0)
        labels = torch.zeros(num_nodes)
        
        # Mark some nodes as buggy (simplified)
        if "bug" in code.lower():
            # Mark 20% of nodes as potentially buggy
            buggy_indices = np.random.choice(num_nodes, size=max(1, num_nodes//5), replace=False)
            labels[buggy_indices] = 1.0
        
        training_data.append((data, labels))
    
    return training_data


if __name__ == "__main__":
    print("Testing Defect Localization Module...")
    
    # Test code
    test_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Potential bug: division by zero if empty list

def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""
    
    # Initialize localizer
    localizer = DefectLocalizer()
    
    # Generate and train on synthetic data
    print("\nGenerating synthetic training data...")
    train_data = generate_synthetic_ast_data(n_samples=20)
    
    print("\nTraining GAT model...")
    localizer.train_model(train_data, epochs=30)
    
    # Localize defects
    print("\nLocalizing defects in test code...")
    results = localizer.localize_defects(test_code, defect_prob=0.8)
    
    print(f"\nFound {len(results['defective_nodes'])} potentially defective nodes")
    print(f"Top 5 suspicious nodes (index, score):")
    for idx, score in results['top_nodes']:
        print(f"  Node {idx}: {score:.4f}")
    
    defective_lines = localizer.get_defective_lines(test_code, defect_prob=0.8)
    print(f"\nPotentially defective lines: {defective_lines}")