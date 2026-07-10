#SNOMED ANALYSIS

#Information for working with 3 relevant SNOMED files

# Concept heirarchy: /SNOMED/sct2_Relationship_Full_US1000124_20240901.txt
# id	effectiveTime	 active	moduleId	sourceId	destinationId	relationshipGroup	typeId	characteristicTypeId	modifierId
# Key columns: active, sourceId,  destinationId , typeId
# Should require active = 1
# typeId should be 'is a' to create tree: 116680003 | is a|
# Use to 1) find children or distant children concepts of Clinical Finding (destinationId = 404684003), 2) build hierarchy tree

# Term to concept: /terminology/sct2_Description_Full-en_US1000124_20240901.txt
# id	effectiveTime	 active	moduleId	conceptId	languageCode	typeId	term	caseSignificanceId
# Key columns: active, conceptId, term
#For definition type id for 900000000000003001 or 'Fully Specified Name'
# Should require active = 1
# Link term text to conceptIds

import pandas as pd
import networkx as nx
import pygraphviz as pgv
import igraph as ig
import math  # Add this at the top of your file if not already imported

print('Reading SNOMED files')
# Read the SNOMED files
relationship_df = pd.read_csv('clinical_trials/SNOMED/sct2_Relationship_Full_US1000124_20240901.txt', sep='\t')
definition_df = pd.read_csv('clinical_trials/SNOMED/sct2_Description_Full-en_US1000124_20240901.txt', sep='\t')
clinical_finding_id = 404684003

print('Filtering active relationships and definitions')
# Filter active relationships and definitions
active_relationships = relationship_df[
    (relationship_df['active'] == 1) & (relationship_df['typeId'] == 116680003)
]
active_definitions = definition_df[
    (definition_df['active'] == 1) & (definition_df['typeId'] == 900000000000003001)
]

print('Length of active relationships: ', len(active_relationships))
print('Length of active definitions: ', len(active_definitions))

# Print the number of distinct conceptIds in active_definitions
print('Number of distinct conceptIds in active_definitions: ', len(active_definitions['conceptId'].unique()))

# Filter active_relationships to only include conceptIds in active_definitions or the clinical finding id
active_relationships = active_relationships[
    (
        active_relationships['destinationId'].isin(active_definitions['conceptId']) |
        (active_relationships['destinationId'] == clinical_finding_id)
    ) &
    (
        active_relationships['sourceId'].isin(active_definitions['conceptId']) |
        (active_relationships['sourceId'] == clinical_finding_id)
    )
]

# Filter to distinct active relationship destination and source
# Filter to include only distinct conceptIds and terms in definitions
active_relationships = active_relationships[['destinationId', 'sourceId']].drop_duplicates()
active_definitions = active_definitions[['conceptId', 'term']].drop_duplicates()

print('Length of active relationships after filtering: ', len(active_relationships))
print('Length of active definitions after filtering: ', len(active_definitions))

print('Creating a dictionary for conceptId to term mapping')
# Create a dictionary for conceptId to term mapping
concept_to_term = dict(zip(active_definitions['conceptId'], active_definitions['term']))

print('Initializing a directed graph')
# Initialize a directed graph
G = nx.DiGraph()
G.add_node(clinical_finding_id)

print('Adding nodes and edges to the graph')
# Add nodes and edges to the graph
for _, row in active_relationships.iterrows():
    source = row['sourceId']
    destination = row['destinationId']
    G.add_edge(destination, source)

print('Pruning the graph to only include descendants of the clinical finding')
# Prune the graph to only include descendants of the clinical finding
descendants = nx.descendants(G, clinical_finding_id)
relevant_nodes = {clinical_finding_id}.union(descendants)
pruned_subgraph = G.subgraph(relevant_nodes)

###############################################################################
##CREATES DATASET WITH ALL DESCENDANTS AND THEIR DEPTH
###############################################################################

# # Function to get all descendants with their depth
# def get_all_descendants_with_depth(graph, start_node):
#     visited = {start_node: 0}
#     current_level = {start_node}
#     depth = 0
#     total_nodes = len(graph.nodes)
#     processed_nodes = 0
    
#     while current_level:
#         next_level = set()
#         for node in current_level:
#             children = set(graph.successors(node)) - visited.keys()
#             for child in children:
#                 visited[child] = depth + 1
#             next_level.update(children)
        
#         processed_nodes += len(current_level)
#         print(f"Processed {processed_nodes}/{total_nodes} nodes at depth {depth}")
        
#         current_level = next_level
#         depth += 1
    
#     return visited

# print('Getting all descendants with their depth')
# descendants_with_depth = get_all_descendants_with_depth(pruned_subgraph, clinical_finding_id)

# # Create a DataFrame from the descendants with depth
# descendants_df = pd.DataFrame([
#     {'conceptId': node, 'description': concept_to_term.get(node, str(node)), 'depth': depth}
#     for node, depth in descendants_with_depth.items()
# ])

# print('Saving descendants with depth to CSV')
# # Output the DataFrame to a CSV file
# descendants_df.to_csv('outputs/descendants_with_depth.csv', index=False)

# # Calculate total number of descendants and max depth
# total_descendants = len(descendants_with_depth) - 1  # Exclude the start node itself
# max_depth = max(descendants_with_depth.values())

total_descendants = 173621
highest_depth = 11
print(f'Total descendants: {total_descendants}, Max depth: {highest_depth}')

################################################################################
###PLOTTING A SAMPLE WITH A MAX DEPTH USING iGraph
################################################################################

# Limit the depth of traversal
def get_descendants_with_depth(graph, start_node, max_depth):
    visited = {start_node}
    current_level = {start_node}
    depth = 0
    while current_level and depth < max_depth:
        next_level = set()
        for node in current_level:
            children = set(graph.successors(node)) - visited
            next_level.update(children)
        visited.update(next_level)
        current_level = next_level
        depth += 1
    return visited

# Function to convert NetworkX graph to iGraph
def convert_nx_to_igraph(nx_graph):
    # Extract edges
    edges = list(nx_graph.edges())
    # Create a mapping from node names to indices
    nodes = list(nx_graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    # Map edges to indices
    edge_tuples = [(node_to_index[src], node_to_index[dst]) for src, dst in edges]
    # Create an undirected iGraph
    g = ig.Graph(edge_tuples, directed=False)
    # Set node names (without labels)
    g.vs['name'] = [str(node) for node in nodes]
    return g

# Set a maximum depth for visualization
max_depth = 11  # Adjust as needed
descendants = get_descendants_with_depth(pruned_subgraph, clinical_finding_id, max_depth)

# Create a subgraph with Clinical Finding and its limited descendants
subgraph_nodes = {clinical_finding_id}.union(descendants)
sampled_subgraph = pruned_subgraph.subgraph(subgraph_nodes).copy()

# Convert to iGraph
igraph_graph = convert_nx_to_igraph(sampled_subgraph)

print('Drawing the graph using iGraph with Reingold-Tilford layout')

# Since the graph is undirected, specify the root node for the layout
root_node_index = igraph_graph.vs.find(name=str(clinical_finding_id)).index

# Use the Reingold-Tilford layout
layout = igraph_graph.layout_reingold_tilford(root=[root_node_index], mode='all')

# Rotate the layout by swapping x and y coordinates
layout_coords = []
for coord in layout.coords:
    # Swap x and y to rotate the layout
    layout_coords.append((coord[1], coord[0]))

# Update the layout with the rotated coordinates
layout = ig.Layout(layout_coords)

# Adjust the bounding box size to accommodate the new orientation
bbox_size = (1800, 1200)  # Swap width and height from (1200, 1800)

# Compute the depth of each node from the root
def compute_node_depths(graph, root_node):
    depths = {root_node: 0}
    queue = [root_node]
    while queue:
        current_node = queue.pop(0)
        current_depth = depths[current_node]
        children = graph.successors(current_node)
        for child in children:
            if child not in depths:
                depths[child] = current_depth + 1
                queue.append(child)
    return depths

# Compute depths for pruned_subgraph
depths_dict = compute_node_depths(pruned_subgraph, clinical_finding_id)

# Add depth attribute to igraph vertices
depths = [depths_dict[int(v['name'])] for v in igraph_graph.vs]
igraph_graph.vs['depth'] = depths

# Decide which nodes to label based on depth or other criteria
label_depths = {0, 1, 8, 9, 10, 11}  # Label nodes at specific depths
labels = []
for v in igraph_graph.vs:
    concept_id = int(v['name'])
    node_depth = v['depth']
    if node_depth in label_depths:
        # Retrieve the term for the concept ID, abbreviate if necessary
        term = concept_to_term.get(concept_id, '')[:50]
        labels.append(term)
    else:
        labels.append('')

# Assign labels and adjust label properties
igraph_graph.vs['label'] = labels
igraph_graph.vs['label_size'] = 4  # Adjust font size as needed
igraph_graph.vs['label_color'] = 'black'  # Set label color
igraph_graph.vs['label_dist'] = 1.5  # Distance from the node

# Update visual style
visual_style = {
    'layout': layout,
    'bbox': bbox_size,
    'margin': 50,
    'vertex_size': 20,
    'vertex_color': 'lightblue',
    'vertex_frame_width': 0,
    'edge_color': 'gray',
    'edge_arrow_size': 0,
    'edge_curved': False,
    'vertex_label': igraph_graph.vs['label'],
    'vertex_label_size': igraph_graph.vs['label_size'],
    'vertex_label_color': igraph_graph.vs['label_color'],
    'vertex_label_dist': igraph_graph.vs['label_dist'],
    'target': 'outputs/snomed_hierarchy_igraph_with_labels_horizontal.svg'
}

# Plot the graph with labels
ig.plot(igraph_graph, **visual_style)

print('Graph with labels saved as SVG in the /outputs folder')

# Also save as PNG
visual_style['target'] = 'outputs/snomed_hierarchy_igraph_with_labels_horizontal.png'
ig.plot(igraph_graph, **visual_style)

print('Graph with labels saved as PNG in the /outputs folder')