"""
PHASE 2: Defect Localization using Graph Attention Network (GAT)
Target: Top-3 localization accuracy ≥ 70%

Based on thesis proposal:
- Parse code to AST
- Augment AST with static features (node type, metrics) and dynamic features (defect probability)
- Convert AST to graph representation
- Train GAT model to predict node-level defectiveness
- Rank nodes and identify top-N bug locations
"""

import ast
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class CodeToGraph:
    """
    Convert Python code to graph representation
    """

    def __init__(self):
        self.node_features = []
        self.node_types = []
        self.edges = []
        self.node_lines = []

    def extract_node_features(self, node, defect_prob=0.5):
        """
        Extract features for each AST node
        Static: node type, complexity, nesting depth
        Dynamic: defect probability from Phase 1
        """
        features = []

        # Node type (one-hot encoded - simplified)
        node_type_map = {
            'FunctionDef': 0, 'ClassDef': 1, 'For': 2, 'While': 3,
            'If': 4, 'Assign': 5, 'Call': 6, 'Return': 7,
            'Import': 8, 'Try': 9, 'ExceptHandler': 10, 'With': 11
        }
        node_type_id = node_type_map.get(node.__class__.__name__, 12)

        # Features: [node_type_id, nesting_depth, defect_prob]
        nesting = self._get_nesting_depth(node)
        features = [node_type_id / 12.0, nesting / 10.0, defect_prob]

        return features

    def _get_nesting_depth(self, node, depth=0):
        """Calculate nesting depth of node"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_nesting_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    def build_graph(self, code, defect_prob=0.5):
        """
        Build graph from code
        Returns: NetworkX graph and node information
        """
        self.node_features = []
        self.node_types = []
        self.edges = []
        self.node_lines = []

        try:
            tree = ast.parse(code)
        except:
            # Return simple graph for unparseable code
            return self._create_default_graph()

        # Build graph structure
        self._traverse_ast(tree, parent_idx=-1, defect_prob=defect_prob)

        # Create NetworkX graph
        G = nx.DiGraph()

        for i, features in enumerate(self.node_features):
            G.add_node(i, features=features, node_type=self.node_types[i], line=self.node_lines[i])

        G.add_edges_from(self.edges)

        return G

    def _traverse_ast(self, node, parent_idx=-1, defect_prob=0.5):
        """Recursively traverse AST and build graph"""
        current_idx = len(self.node_features)

        # Extract features
        features = self.extract_node_features(node, defect_prob)
        self.node_features.append(features)
        self.node_types.append(node.__class__.__name__)

        # Get line number
        line = getattr(node, 'lineno', -1)
        self.node_lines.append(line)

        # Add edge from parent
        if parent_idx >= 0:
            self.edges.append((parent_idx, current_idx))

        # Traverse children
        for child in ast.iter_child_nodes(node):
            self._traverse_ast(child, current_idx, defect_prob)

    def _create_default_graph(self):
        """Create simple default graph for error cases"""
        G = nx.DiGraph()
        G.add_node(0, features=[0.0, 0.0, 0.5], node_type='Module', line=1)
        return G

    def to_pytorch_geometric(self, G):
        """
        Convert NetworkX graph to PyTorch Geometric Data
        """
        # Node features
        node_features = []
        for node in G.nodes():
            node_features.append(G.nodes[node]['features'])

        x = torch.tensor(node_features, dtype=torch.float)

        # Edges
        edge_index = []
        if G.edges():
            edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        return data


class GATDefectLocalizer(nn.Module):
    """
    Graph Attention Network for Defect Localization
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4):
        super(GATDefectLocalizer, self).__init__()

        print("\n" + "="*70)
        print("GAT MODEL STRUCTURE")
        print("="*70)

        # GAT layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.6)

        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)

        print(f"Input Layer:   {input_dim} features")
        print(f"GAT Layer 1:   {hidden_dim} hidden units, {num_heads} attention heads")
        print(f"GAT Layer 2:   {hidden_dim} hidden units, 1 attention head")
        print(f"Output Layer:  1 unit (defectiveness score)")
        print(f"Total Params:  {sum(p.numel() for p in self.parameters())}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Output
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x.squeeze()


class DefectLocalizer:
    """
    Phase 2: GAT-based Defect Localization
    """

    def __init__(self, hidden_dim=64, num_heads=4):
        self.code_to_graph = CodeToGraph()
        self.model = GATDefectLocalizer(input_dim=3, hidden_dim=hidden_dim, num_heads=num_heads)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def localize_defects(self, code, defect_prob=0.5, top_n=3):
        """
        Localize defects in code
        Returns top-N most suspicious nodes/lines
        """
        print("\n" + "="*70)
        print("PHASE 2: DEFECT LOCALIZATION")
        print("="*70)

        # Build graph
        print("\nStep 1: Code Parsing & Graph Construction...")
        G = self.code_to_graph.build_graph(code, defect_prob)
        print(f"  Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # Convert to PyTorch Geometric
        print("\nStep 2: Converting to Graph Representation...")
        data = self.code_to_graph.to_pytorch_geometric(G)
        data = data.to(self.device)
        print(f"  Node features shape: {data.x.shape}")
        print(f"  Edge index shape: {data.edge_index.shape}")

        # Get predictions
        print("\nStep 3: GAT Model Inference...")
        self.model.eval()
        with torch.no_grad():
            scores = self.model(data)

        scores_np = scores.cpu().numpy()

        # Rank nodes
        ranked_indices = np.argsort(scores_np)[::-1]

        # Get top-N nodes with line numbers
        print(f"\nStep 4: Identifying Top-{top_n} Suspicious Nodes...")

        suspicious_nodes = []
        for idx in ranked_indices[:min(top_n, len(ranked_indices))]:
            node_info = {
                'node_id': int(idx),
                'score': float(scores_np[idx]),
                'node_type': self.code_to_graph.node_types[idx] if idx < len(self.code_to_graph.node_types) else 'Unknown',
                'line': int(self.code_to_graph.node_lines[idx]) if idx < len(self.code_to_graph.node_lines) else -1
            }
            suspicious_nodes.append(node_info)

            print(f"  Rank {len(suspicious_nodes)}: Line {node_info['line']}, "
                  f"Type={node_info['node_type']}, Score={node_info['score']:.4f}")

        # Visualize graph (save to file)
        self._visualize_graph(G, scores_np, suspicious_nodes)

        results = {
            'graph': G,
            'scores': scores_np,
            'suspicious_nodes': suspicious_nodes,
            'top_lines': [n['line'] for n in suspicious_nodes if n['line'] > 0]
        }

        print(f"\n✓ Localization complete!")
        print(f"  Top suspicious lines: {results['top_lines']}")

        return results

    def _visualize_graph(self, G, scores, suspicious_nodes, filename='phase2_graph.png'):
        """Visualize the code graph with scores"""
        try:
            plt.figure(figsize=(12, 8))

            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Node colors based on scores
            node_colors = [scores[i] if i < len(scores) else 0.5 for i in G.nodes()]

            # Draw
            nx.draw(G, pos,
                   node_color=node_colors,
                   cmap=plt.cm.Reds,
                   node_size=500,
                   with_labels=True,
                   font_size=8,
                   arrows=True,
                   edge_color='gray',
                   alpha=0.7)

            plt.title('Phase 2: Code Graph with Defectiveness Scores')
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), label='Defectiveness Score')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Graph visualization saved: {filename}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")

    def evaluate_localization(self, predictions, ground_truth, top_k=3):
        """
        Evaluate localization accuracy
        Target: Top-3 accuracy ≥ 70%
        """
        print("\n" + "="*70)
        print("LOCALIZATION EVALUATION")
        print("="*70)

        correct = 0
        total = len(ground_truth)

        for pred_lines, true_lines in zip(predictions, ground_truth):
            # Check if any true bug is in top-k predictions
            if any(line in pred_lines[:top_k] for line in true_lines):
                correct += 1

        accuracy = correct / total if total > 0 else 0

        print(f"\nTop-{top_k} Localization Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Correct: {correct}/{total}")
        print(f"  Target: ≥ 70%")
        print(f"  Status: {'✓ PASS' if accuracy >= 0.70 else '✗ FAIL'}")

        return accuracy


def demo_phase2():
    """
    Demonstration of Phase 2: Defect Localization
    """
    print("\n" + "="*70)
    print("PHASE 2: DEFECT LOCALIZATION - DEMONSTRATION")
    print("="*70)

    # Sample buggy code
    buggy_code = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Bug: no zero check

def process_data(data):
    results = []
    for i in range(len(data)):
        value = data[i + 1]  # Bug: index out of range
        results.append(value)
    return results

def risky_operation(x):
    try:
        result = 10 / x  # Bug: potential division by zero
        return result
    except:  # Bug: bare except
        pass
"""

    print("\nTest Code:")
    print(buggy_code)

    # Initialize localizer
    localizer = DefectLocalizer(hidden_dim=64, num_heads=4)

    # Localize defects (with high defect probability from Phase 1)
    results = localizer.localize_defects(buggy_code, defect_prob=0.85, top_n=5)

    print("\n" + "="*70)
    print("✓ PHASE 2 COMPLETE")
    print("="*70)

    return localizer, results


if __name__ == "__main__":
    localizer, results = demo_phase2()
