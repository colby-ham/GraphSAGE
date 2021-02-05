import networkx as nx
from networkx.readwrite import json_graph
import json

def read_edges(path, split):
    edges = []
    with open(path, 'r') as f:
        for l in f:
            tokens = l.split()
            rel = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            if split != 'train':
                label = tokens[3]
                if label == '0':
                    #print("False edge... continuing")
                    continue

            edge = (int(n1), int(n2))
            edges.append(edge)
    return edges


def main():
    # Load in kgat data
    data_dir = "/people/hamc649/recommendation/kgat_data"
    #data_dir = '/Users/d3m432/git/GraphSAGE/sutanay_graphsage/kgat_data'
    dataset = 'last-fm'
    dataset_dir = f"{data_dir}/{dataset}"
    output_dir = "/people/hamc649/recommendation/GraphSAGE/example_data"
    #output_dir = '/Users/d3m432/git/GraphSAGE/sutanay_graphsage/GraphSAGE/example_data'
    train_path = f"{dataset_dir}/train.txt"
    valid_path = f"{dataset_dir}/valid.txt" # exclude false edges
    kg_path = f"{dataset_dir}/kg_final.txt"
    node_type_vocab_path = f"{dataset_dir}/node_type_vocab.txt"

    # read in edges
    edges = list()
    train_edges = read_edges(train_path, 'train')
    valid_edges = read_edges(valid_path, 'valid')
    kg_edges = read_edges(kg_path, 'train')
    
    edges.extend(train_edges)
    edges.extend(valid_edges)
    edges.extend(kg_edges)

    # Get all node ids from edges
    node_ids = set()
    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]
        node_ids.add(n1)
        node_ids.add(n2)
    max_node_id = max(node_ids)
    min_node_id = min(node_ids)

    # Read in maps
    with open(node_type_vocab_path, 'r') as f:
        node_type_vocab = json.load(f)
    
    # convert map to one hot
    node_type_vocab_oh = dict()
    for k in node_type_vocab:
        v = node_type_vocab[k]
        if v == 'item':
            nv = [1, 0, 0]
        elif v == 'entity':
            nv = [0, 1, 0]
        else:
            # user
            nv = [0, 0, 1]
        node_type_vocab_oh[k] = nv


    # Transform kgat_data into networkx graph
    G = nx.from_edgelist(edges)
    # We don't need to hardcode max_node_id
    all_node_ids = {i for i in range(max_node_id + 1)}
    remaining_node_ids = all_node_ids.difference(node_ids)

    #import pdb;pdb.set_trace()

    #for i in range(max_node_id + 1):
    #    G.add_node(i)
    for node_id in remaining_node_ids:
        G.add_node(node_id)

    #import pdb;pdb.set_trace()
    nx.set_node_attributes(G, 'val', False)
    nx.set_node_attributes(G, 'test', False)
    #import pdb;pdb.set_trace()
        
    # Dump necessary networkx files
    # 1. Dump graph
    output_path = f"{output_dir}/{dataset}-G.json"
    data = json_graph.node_link_data(G)
    #import pdb;pdb.set_trace()
    with open(output_path, 'w') as f:
        json.dump(data, f)

    # 2. dump class map
    # for each node, add class to dictionary
    class_map = dict()
    nodes = G.nodes()
    print(f"Num nodes: {G.number_of_nodes()}, Num edges: {G.number_of_edges()}")
    for node in nodes:
        #class_map[str(node)] = [1, 0, 0]
        class_map[str(node)] = node_type_vocab_oh[str(node)]
    class_map_path = f"{output_dir}/{dataset}-class_map.json"
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f)

    # 3. dump id map
    id_map = {str(i): i for i in range(len(nodes))}
    id_map_path = f"{output_dir}/{dataset}-id_map.json"
    with open(id_map_path, 'w') as f:
        json.dump(id_map, f)





if __name__ == '__main__':
    main()
