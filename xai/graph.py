import networkx as nx
import matplotlib.pyplot as plt
import torch

class DeeppyGraph:
    def __init__(self, layers, node_values=None, edge_values=None):
        self.graph = nx.DiGraph()
        self.positions = {}
        self.layers = layers
        self.node_values = node_values  # can be torch tensor
        self.edge_values = edge_values  # can be torch tensor
        self._build_graph()

    def _build_graph(self):
        """Builds the network layout and connections."""
        layer_count = len(self.layers)
        x_spacing = 3
        max_neurons = max(self.layers)
        total_height = max_neurons * 0.5

        for layer_idx, num_nodes in enumerate(self.layers):
            layer_type = (
                "input" if layer_idx == 0 else
                "output" if layer_idx == layer_count - 1 else
                "hidden"
            )
            
            y_spacing = total_height / (num_nodes if num_nodes > 1 else 1)
            y_offset = (num_nodes - 1) / 2 * y_spacing

            for i in range(num_nodes):
                node_name = f"{layer_type}_{layer_idx}_{i}"
                self.graph.add_node(node_name, layer=layer_type, layer_idx=layer_idx)
                x = layer_idx * x_spacing
                y = -(i * y_spacing) + y_offset
                self.positions[node_name] = (x, y)

        # fully connect each pair of consecutive layers
        for l in range(layer_count - 1):
            src_nodes = [n for n in self.graph if self.graph.nodes[n]['layer_idx'] == l]
            dst_nodes = [n for n in self.graph if self.graph.nodes[n]['layer_idx'] == l + 1]
            for src in src_nodes:
                for dst in dst_nodes:
                    self.graph.add_edge(src, dst)

    def _get_colormap(self, values, mode='color', base_color='gray'):
        """Map values to colors or alpha."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        if not values.any() and mode=='color':
            return None, None

        if mode == 'color':
            vmin, vmax = min(values), max(values)
            abs_max = max(abs(vmin), abs(vmax)) or 1e-9
            norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cmap = cm.get_cmap('bwr')  # blue-white-red
            return cmap(norm(values)), None
        elif mode == 'alpha':
            # normalize to 0-1 for alpha
            min_val, max_val = values.min(), values.max()
            alpha = (values - min_val) / ((max_val - min_val) or 1e-9)
            color = np.array(colors.to_rgba(base_color))
            rgba = np.zeros((len(values), 4))
            rgba[:, :3] = color[:3]  # RGB
            rgba[:, 3] = alpha       # alpha
            return rgba, None
        else:
            raise ValueError("Mode must be 'color' or 'alpha'")

    def draw(self, figsize=(12, 8),
             node_mode='color', node_color='lightgray',
             edge_mode='color', edge_color='gray'):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Nodes
        node_vals = self.node_values if self.node_values is not None else torch.zeros(len(self.graph))
        node_colors, _ = self._get_colormap(node_vals, mode=node_mode, base_color=node_color)
        
        # Edges
        edge_vals = self.edge_values if self.edge_values is not None else torch.zeros(len(self.graph.edges))
        edge_colors, _ = self._get_colormap(edge_vals, mode=edge_mode, base_color=edge_color)
        
        # Draw edges first
        nx.draw_networkx_edges(
            self.graph, pos=self.positions, ax=ax,
            arrows=False, edge_color=edge_colors, width=0.5, alpha=0.7 if edge_mode=='color' else 1.0
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos=self.positions, ax=ax,
            node_color=node_colors, node_size=150,
            edgecolors='black', linewidths=0.3
        )
        
        ax.set_title(f"DeeppyGraph Visualization: {self.layers}", fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
