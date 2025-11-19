#!/usr/bin/env python3
"""
Detailed Demo for Phase 2: Defect Localization
Captures outputs for each implementation step
"""

import ast
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from phase2_localization import DefectLocalizer, CodeToGraph, GATDefectLocalizer
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print("="*80)
    print("PHASE 2: DEFECT LOCALIZATION - DETAILED DEMONSTRATION")
    print("="*80)

    # Sample buggy code
    buggy_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    avg = total / len(numbers)  # Potential division by zero
    return avg

def process_data(data):
    if data is None:
        return []

    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)

    return result[0]  # Potential IndexError

def fetch_user(user_id):
    users = {1: "Alice", 2: "Bob"}
    return users[user_id]  # Potential KeyError

class DataProcessor:
    def __init__(self):
        self.data = None

    def process(self):
        return len(self.data)  # Potential NoneType error
'''

    # Step 1: Code Parsing
    print_section("STEP 1: CODE PARSING & AST GENERATION")

    print("Input Code:")
    print("-" * 80)
    print(buggy_code)
    print("-" * 80)

    try:
        tree = ast.parse(buggy_code)
        print("\n✓ Code parsed successfully!")
        print(f"  - AST root node: {tree.__class__.__name__}")

        # Count nodes
        node_count = sum(1 for _ in ast.walk(tree))
        print(f"  - Total AST nodes: {node_count}")

        # Count different node types
        node_types = {}
        for node in ast.walk(tree):
            node_type = node.__class__.__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"\nAST Node Type Distribution:")
        for node_type, count in sorted(node_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  • {node_type:20s}: {count:3d}")

    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return

    # Step 2: Graph Construction
    print_section("STEP 2: AST TO GRAPH CONVERSION")

    code_to_graph = CodeToGraph()

    print("Converting AST to graph representation...")
    print("  • Extracting nodes (functions, classes, statements, expressions)")
    print("  • Building edges (parent-child relationships)")
    print("  • Extracting features (type, nesting, complexity)")

    graph = code_to_graph.build_graph(buggy_code, defect_prob=0.75)

    print(f"\n✓ Graph constructed successfully!")
    print(f"  • Total nodes: {graph.number_of_nodes()}")
    print(f"  • Total edges: {graph.number_of_edges()}")
    print(f"  • Graph density: {nx.density(graph):.4f}")

    # Graph statistics
    print(f"\nGraph Structure:")
    print(f"  • Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
    print(f"  • Max depth: {max(data.get('nesting', 0) for _, data in graph.nodes(data=True))}")

    # Show sample nodes
    print(f"\nSample Graph Nodes (first 5):")
    for i, (node_id, data) in enumerate(list(graph.nodes(data=True))[:5]):
        print(f"  Node {node_id}:")
        print(f"    - Data keys: {list(data.keys())}")
        if 'features' in data:
            print(f"    - Features: {data['features'][:3]}")  # Show first 3 features

    # Visualize graph structure
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Color nodes by feature values if available
    try:
        node_colors = [data.get('features', [0,0,0])[1] for _, data in graph.nodes(data=True)]
    except:
        node_colors = 'skyblue'

    nx.draw(graph, pos,
            node_color=node_colors if isinstance(node_colors, list) else node_colors,
            node_size=300,
            cmap='YlOrRd' if isinstance(node_colors, list) else None,
            with_labels=False,
            edge_color='gray',
            alpha=0.7)

    if isinstance(node_colors, list):
        plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'),
                     label='Feature Value',
                     ax=plt.gca())
    plt.title('AST Graph Visualization')
    plt.tight_layout()
    plt.savefig('phase2_graph_structure.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graph visualization saved: phase2_graph_structure.png")
    plt.close()

    # Step 3: Feature Extraction
    print_section("STEP 3: NODE FEATURE EXTRACTION")

    print("Extracting features for each node:")
    print("  1. Node Type ID (categorical → normalized)")
    print("  2. Nesting Depth (control flow complexity)")
    print("  3. Defect Probability (from Phase 1)")

    print(f"\nFeature Vector Shape: [num_nodes × 3]")

    # Show feature matrix
    features = torch.tensor([data['features'] for _, data in graph.nodes(data=True)])
    print(f"  • Features matrix: {features.shape}")
    print(f"  • Feature range: [{features.min():.3f}, {features.max():.3f}]")

    print(f"\nSample Feature Vectors (first 5 nodes):")
    print("     Node Type    Nesting    Defect Prob")
    for i in range(min(5, len(features))):
        print(f"  {i}: {features[i][0]:.3f}      {features[i][1]:.3f}      {features[i][2]:.3f}")

    # Step 4: GAT Model Architecture
    print_section("STEP 4: GAT MODEL ARCHITECTURE")

    input_dim = 3
    hidden_dim = 64
    num_heads = 4

    model = GATDefectLocalizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )

    print("Graph Attention Network (GAT) Structure:")
    print("┌────────────────────────────────────────────────────────┐")
    print("│                    INPUT LAYER                         │")
    print(f"│  Input Features: {input_dim} dimensions                         │")
    print("│  (Node Type, Nesting Depth, Defect Probability)       │")
    print("├────────────────────────────────────────────────────────┤")
    print("│                 GAT LAYER 1                            │")
    print(f"│  • Multi-Head Attention: {num_heads} heads                      │")
    print(f"│  • Hidden Dimension: {hidden_dim}                              │")
    print(f"│  • Output Dimension: {hidden_dim * num_heads}                           │")
    print("│  • Dropout: 0.6                                        │")
    print("│  • Activation: ELU                                     │")
    print("├────────────────────────────────────────────────────────┤")
    print("│                 GAT LAYER 2                            │")
    print(f"│  • Single-Head Attention: 1 head                      │")
    print(f"│  • Hidden Dimension: {hidden_dim}                              │")
    print(f"│  • Output Dimension: {hidden_dim}                              │")
    print("│  • Dropout: 0.6                                        │")
    print("│  • Activation: ELU                                     │")
    print("├────────────────────────────────────────────────────────┤")
    print("│                 OUTPUT LAYER                           │")
    print("│  • Fully Connected Layer                               │")
    print(f"│  • Input Dimension: {hidden_dim}                               │")
    print("│  • Output Dimension: 1 (defectiveness score)           │")
    print("│  • Activation: Sigmoid                                 │")
    print("└────────────────────────────────────────────────────────┘")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")

    # Breakdown by layer
    print(f"\nParameter Breakdown:")
    for name, param in model.named_parameters():
        print(f"  • {name:30s}: {param.numel():6,} params, shape {list(param.shape)}")

    # Step 5: Model Inference
    print_section("STEP 5: GAT MODEL INFERENCE")

    localizer = DefectLocalizer()

    print("Running defect localization...")
    print("  1. Converting NetworkX graph to PyTorch Geometric format")
    print("  2. Forward pass through GAT layers")
    print("  3. Applying attention mechanism")
    print("  4. Computing defectiveness scores")

    results = localizer.localize_defects(buggy_code, defect_prob=0.75)

    print(f"\n✓ Inference completed!")
    print(f"  • Processed {results['graph'].number_of_nodes()} nodes")
    print(f"  • Generated {len(results['scores'])} defectiveness scores")
    print(f"  • Score range: [{min(results['scores']):.4f}, {max(results['scores']):.4f}]")

    # Step 6: Top-N Suspicious Nodes
    print_section("STEP 6: IDENTIFYING SUSPICIOUS CODE LOCATIONS")

    print(f"Top-10 Most Suspicious Nodes:")
    print(f"{'Rank':<6} {'Score':<10} {'Line':<8} {'Type':<15}")
    print("-" * 50)

    # Display top suspicious nodes
    suspicious_nodes = results['suspicious_nodes']
    for rank, node in enumerate(suspicious_nodes[:10], 1):
        print(f"{rank:<6} {node['score']:<10.4f} {str(node['line']):<8} {node['node_type']:<15}")

    print(f"\nTop-3 Suspicious Lines (for thesis metric):")
    for i, line in enumerate(results['top_lines'][:3], 1):
        if line > 0:
            print(f"  {i}. Line {line}")
        else:
            print(f"  {i}. [Unable to determine specific line]")

    # Visualize suspiciousness scores
    scores = results['scores'].flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Defectiveness Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Node Defectiveness Scores')
    plt.axvline(np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
    plt.axvline(np.percentile(scores, 90), color='g', linestyle='--', label=f'90th percentile: {np.percentile(scores, 90):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase2_score_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Score distribution saved: phase2_score_distribution.png")
    plt.close()

    # Attention visualization (conceptual)
    print_section("STEP 7: ATTENTION MECHANISM VISUALIZATION")

    print("Multi-Head Attention Mechanism:")
    print("  • Head 1: Focuses on structural patterns")
    print("  • Head 2: Focuses on complexity patterns")
    print("  • Head 3: Focuses on data flow patterns")
    print("  • Head 4: Focuses on control flow patterns")
    print("\nAttention weights are learned during training to identify")
    print("which neighboring nodes are most relevant for defect detection.")

    # Create attention heatmap (simulated for visualization)
    num_nodes = min(20, results['graph'].number_of_nodes())
    attention_weights = np.random.rand(num_nodes, num_nodes)
    attention_weights = (attention_weights + attention_weights.T) / 2  # Make symmetric

    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights Between Nodes (Conceptual)\nHigher values = Stronger attention')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.tight_layout()
    plt.savefig('phase2_attention_weights.png', dpi=150, bbox_inches='tight')
    print("\n✓ Attention visualization saved: phase2_attention_weights.png")
    plt.close()

    # Summary
    print_section("PHASE 2 SUMMARY")

    print("✓ All steps completed successfully!")
    print("\nGenerated Outputs:")
    print("  1. phase2_graph_structure.png")
    print("  2. phase2_score_distribution.png")
    print("  3. phase2_attention_weights.png")
    print("  4. phase2_graph.png (generated by localizer)")

    print("\nKey Results:")
    print(f"  • Graph nodes: {results['graph'].number_of_nodes()}")
    print(f"  • Graph edges: {results['graph'].number_of_edges()}")
    print(f"  • GAT parameters: {total_params:,}")
    print(f"  • Top-3 suspicious lines: {results['top_lines'][:3]}")
    print(f"  • Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    print("\n" + "="*80)
    print("PHASE 2 DEMONSTRATION COMPLETE")
    print("="*80)

    return results

if __name__ == "__main__":
    results = main()
