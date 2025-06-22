import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community as community_louvain

# Configuration
data_folder = 'data'
output_dir = 'visualizations'
# Minimum total edge weight or occurrence count to keep a node
MIN_TOTAL_WEIGHT = 100
# Alias map for merging variant names
alias_map = {
    'ronald': 'ron',
    'ronald weasley': 'ron',
    'miss granger': 'hermione',
    'hermione granger': 'hermione',
    'harry potter': 'harry',
    'mr. potter': 'harry',
    'albus': 'dumbledore', 
    'albus dumbledore': 'dumbledore',
    'rubeus hagrid': 'hagrid',
    'rubeus': 'hagrid',
    'dark lord': 'voldemort',
    'lord': 'voldemort',
    'uncle': 'vernon',
    'draco malfoy' : 'malfoy',
    'gilderoy lockhart': 'lockhart',
    'sirus black': 'sirius',
    'severus snape': 'snape',
    'severus': 'snape',
    'tom riddle': 'voldemort',
    'marvolo': 'voldemort'
    # Add other variants as needed
}

# Stopplist: characters to exclude from visualization
stopplist = {'mr.', 'mrs.', 'miss', 'dr.', 'sir', 'madam', 'narrator', 'unknown', 'the', 'page', 'forest', 'voice', 'will', 'dungeon', 
              'staring', 'two', 'boy', 'three', 'third', 'able', 'rich', 'magic', 'dark', 'father', 'potions', 'witch', 'fat', 'little',
              'mark', 'st', 'hall', 'house', 'death', 'face', 'bloody', 'ice', 'quiditch', 'ceremony', 'serpent', 'second', 'ghost', 'kit',
              'keeper', 'secret', 'angry', 'ministry', 'curious', 'night', 'ancient', 'bane', 'old', 'tall', 'nope' 'young', 'leaky', 'evil',
              'goblin', 'entrance', 'squat,', 'circe', 'gloomy', 'knight', 'norwegian', 'hope', 'first', 'music', 'red', 'platform', 'day',
              'fallen', 'wizard', 'group', 'h.', 'g.', 'newspaper', 'crowd', 'owner', 'red-haired', 'm.', 'magical', 'holy', 'paper', 'mother',
              'hogwarts', 'stunned', 'man', 'forbidden', 'rose', 'broom', 'black', 'brothers'}  # extend as needed



# Find all relationship CSVs in data_folder
relationship_files = [
    entry.name for entry in os.scandir(data_folder)
    if entry.is_file() and 'relationship' in entry.name.lower() and entry.name.endswith('.csv')
]

# Normalize and map names to lowercase and apply alias_map
def normalize_df(df):
    df['source'] = df['source'].str.lower().map(lambda x: alias_map.get(x, x))
    df['target'] = df['target'].str.lower().map(lambda x: alias_map.get(x, x))
    return df

# Filter out weakly connected nodes
def filter_weak_nodes(df, min_weight=MIN_TOTAL_WEIGHT):
    # compute total weight per node
    weights = pd.concat([
        df.groupby('source')['value'].sum(),
        df.groupby('target')['value'].sum()
    ]).groupby(level=0).sum()
    strong_nodes = set(weights[weights >= min_weight].index)
    # keep edges where both ends are strong nodes
    return df[df['source'].isin(strong_nodes) & df['target'].isin(strong_nodes)]

# Build and style network with node sizing by weight
def create_network_from_df(df, width, height, enable_physics=False):
    # 1) NetworkX-Graph aufbauen
    G = nx.from_pandas_edgelist(
        df, source='source', target='target', edge_attr='value', create_using=nx.Graph()
    )
    part = community_louvain.best_partition(G)
    nx.set_node_attributes(G, part, 'group')
    for node in G.nodes():
        total_weight = sum(d['value'] for _, _, d in G.edges(node, data=True))
        size = 10 + total_weight/50
        G.nodes[node]['size'] = size

    # 2) PyVis-Network erzeugen und Graph importieren
    net = Network(
        width=f'{width}px',
        height=f'{height}px',
        bgcolor="#000000",
        font_color='white',    # Default-Font für Buttons, etc.
        directed=False
    )
    net.from_nx(G)

    # 3) Für jede Node die font-Eigenschaften setzen
    for n in net.nodes:
        n['font'] = {
            'size': n['size'],  # Schriftgröße = Knotengröße
            'color': 'white'     # Schriftfarbe explizit weiß
        }

    # 4) Physics-Buttons
    if enable_physics:
        net.show_buttons(filter_=['physics'])
    else:
        net.toggle_physics(False)

    return net


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate and save visualizations
for rel_file in relationship_files:
    csv_path = os.path.join(data_folder, rel_file)
    df = pd.read_csv(csv_path)

    # Normalize names and remove stoplisted
    df = normalize_df(df)
    df = df[~df['source'].isin(stopplist) & ~df['target'].isin(stopplist)]
    # Filter weakly connected nodes
    df = filter_weak_nodes(df)

    # Derive output name and sizing
    base = os.path.splitext(rel_file)[0]
    html_file = f"{base}.html"
    width, height = (2000, 1000)
    if 'b4' in base.lower(): width = 2000

    enable_physics = True
    net = create_network_from_df(df, width, height, enable_physics)

    out_path = os.path.join(output_dir, html_file)
    net.write_html(out_path, notebook=False)
    print(f"Saved network visualization: {out_path}")
