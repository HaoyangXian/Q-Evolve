# build Q data to train qnet
import argparse
import json
import os
root_dir = os.getcwd()
from pathlib import Path
import sys
sys.path.append(root_dir)
sys.path.append(str(Path(root_dir).parent))
import pickle
from tqdm import tqdm

class TreeNode:
    _id_counter = 0
    def __init__(self, state, action,reward=0, q_value=0):
        self.id = TreeNode._get_next_id() 
        self.state = state # list of messages
        self.action = action # action of the current node
        self.reward = reward
        self.q_value = q_value
        self.children = []
        
    @classmethod
    def _get_next_id(cls):
        current_id = cls._id_counter
        cls._id_counter += 1
        return current_id
    def add_child(self, child_node):
        self.children.append(child_node)

    def update_q_values(self, gamma=0.9):
        if not self.children:
            self.q_value = self.reward if self.reward is not None else 0
        else:
            for child in self.children:
                child.update_q_values(gamma)
            self.q_value = self.reward + gamma *max(child.q_value for child in self.children)
            # self.q_value = self.reward + gamma * max(child.q_value for child in self.children)

    def update_rewards(self):
        if not self.children:
            if self.reward is None:
                self.reward = 0
            return [self.reward]
        else:
            leaf_rewards = []
            for child in self.children:
                leaf_rewards.extend(child.update_rewards())
            self.reward = sum(leaf_rewards) / len(leaf_rewards)
            return leaf_rewards
        
    def print_info(self):
        print(f"State: {self.state}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}")
        print(f"Q-value: {self.q_value}")

    def visualize(self, level=0):
        indent = "    " * level
        # import ipdb; ipdb.set_trace()
        #print(f"Level {level}, {indent}Node(State: {self.state}, Action: {self.action}, Reward: {self.reward}, Q-value: {self.q_value})")
        for child in self.children:
            child.visualize(level + 1)

    def visualize_graph(self, G=None, pos=None, parent_name=None, level=0, horiz_spacing=1):
        if G is None:
            G = nx.DiGraph()
            pos = {}
        
        node_name = str(id(self))
        short_label = (self.action[:5] + '...') if len(self.action) > 5 else self.action
        short_label = ""
        node_label = f"{short_label}\nR: {round(self.reward,2)}\nQ: {round(self.q_value,2)}"
        G.add_node(node_name, label=node_label)
        
        # Compute horizontal position by counting nodes at this level
        if level not in pos:
            pos[level] = 0  # Initialize counter for this level
        x = pos[level] * horiz_spacing
        pos[node_name] = (x, -level)
        pos[level] += 1  # Increment counter at this level

        if parent_name:
            G.add_edge(parent_name, node_name)
        
        for child in self.children:
            child.visualize_graph(G, pos, node_name, level + 1, horiz_spacing)

        return G, {node: coords for node, coords in pos.items() if isinstance(coords, tuple)}  # Filter to keep only coordinates
    
    def child_num(self):
        return len(self.children)


def combine_jsonl(input_files, output_file):
    combined_data = []
    for file in input_files:
        data = json.load(open(file, 'r'))
        print(len(data))
        combined_data.extend(data)

    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)


def combine_pkl(files, output_file):
    combined_data = []
    for file in files:
        with open(file, 'rb') as f:
            while True:
                try:
                    tree = pickle.load(f)
                    combined_data.append(tree)
                except EOFError:
                    break
            # combined_data = []
        print(len(combined_data))
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

def load_trees(file_path):
    trees = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                tree = pickle.load(f)
                trees.append(tree)
            except EOFError:
                break
    return trees

def post_process_tree(root_node):
    assert "Root" in root_node.action and len(root_node.children)>1
    # search_nodes= [child for child in node.children if "search[" in child.action and "]" in child.action]

    import re
    pattern = r'\[(.*?)\]'
    action_value = root_node.children[-1].action['value']
    match = re.search(pattern, action_value)
    if match:
        sft_search_content = match.group(1)
    merged_idx = []
    node_idx = 0
    for idx in range(len(root_node.children)-1):
        node = root_node.children[node_idx]
        action_value = node.action['value']
        match = re.search(pattern, action_value)
        if match:
            content = match.group(1)
            if content == sft_search_content: # merge sft node with explored node which search for the same content with sft
                for child in node.children:
                    root_node.children[-1].add_child(child)
                root_node.children.pop(node_idx) 
                print("merge sft node with explored node which search for the same content with sft")
                node_idx -=1
        node_idx += 1
    
    def del_repeated_child(node):
        node_action_list = [child.action['value'] for child in node.children]
        seen_actions = set()
        unique_children = []

        for child in node.children:
            action_value = child.action['value'].split("Action:")[-1].strip()
            if action_value not in seen_actions:
                seen_actions.add(action_value)  
                unique_children.append(child)
            node.children = unique_children
        for child in node.children:
            del_repeated_child(child)
    del_repeated_child(root_node)
    return root_node

def collect_q_data_from_a_tree(node):
    data = []
    def recurse(node):
        if node.action and node.q_value is not None :

            if not "Root" in node.action:
                #print("node.action",node.action)
                assert isinstance(node.action, dict) and node.action['from']=='gpt'
                traj = node.state + [node.action]
   
            entry = {
                'conversations': traj,
                'label': node.q_value,
                'reward': node.reward,
                # 'next_state': [child.state[-1] if child.state else None for child in node.children],
            }
            data.append(entry)
        
        for child in node.children:
            recurse(child)
    # start from children
    for child in node.children:
        recurse(child)    

    return data

def collect_r_data_from_a_tree(node):
    data = []
    def recurse(node):
        if node.action and node.q_value is not None :

            if not "Root" in node.action:
                #print("node.action",node.action)
                assert isinstance(node.action, dict) and node.action['from']=='gpt'
                traj = node.state + [node.action]
            entry = {
                'conversations': traj,
                'label': node.reward,
            }
            data.append(entry)
        
        for child in node.children:
            recurse(child)
    # start from children
    for child in node.children:
        recurse(child)    

    return data

def normalize_data(data, feature_key):
    values = [entry[feature_key] for entry in data if feature_key in entry]
    min_val, max_val = min(values), max(values)
    # print("min_val",min_val)
    # print("max_val",max_val)
    for entry in data:
        if feature_key in entry:
            entry[feature_key] = (entry[feature_key] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return data

        
def update_depth_q_values(node,height,gamma=0.9):
    if not node.children:
        reward = node.reward if node.reward is not None else 0
        node.q_value = reward
        if height>1:
            for i in range(1,height):
                node.q_value += gamma**i*reward
    else:
        for child in node.children:
            update_depth_q_values(child,height-1,gamma=gamma)
        node.q_value = node.reward + gamma * max([child.q_value for child in node.children])
    
def collect_q_data_from_trees_unpruned(tree_pth,save_q_pth):
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    q_data_ls = []
    for tree in tqdm(trees[0]):
        for t in tree:
            id = t['id']
            print("id",id)
            # post_processed_t = post_process_tree(t['tree'])
            post_processed_t = t['tree']
            post_processed_t.update_rewards()
            update_depth_q_values(post_processed_t,height=5,gamma=0.9)
            q_data = collect_q_data_from_a_tree(post_processed_t)
            q_data = normalize_data(q_data, 'label')
            q_data = [{'id': id, **entry} for entry in q_data]
            q_data_ls.extend(q_data)
    with open(save_q_pth, 'w') as f:
        json.dump(q_data_ls, f, indent=4)

def collect_vanilla_q_data_from_trees_unpruned(tree_pth, save_q_pth, upper_num=300):
    import random
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    q_data_ls = []
    def update_vanilla_rewards(node):
        if len(node.children) > 0:
            node.reward = 0
            for child in node.children:
                update_vanilla_rewards(child)
        else:
            node.reward = node.reward
    num_tree = len(trees[0])
    for n, tree in tqdm(enumerate(trees[0])):
        if n == num_tree-1:
            break
        for t in tree:
            id = t['id']
            # post_processed_t = post_process_tree(t['tree'])
            # post_processed_t.update_rewards()
            post_processed_t = t['tree']
            update_vanilla_rewards(post_processed_t)
            # import ipdb; ipdb.set_trace()
            post_processed_t.update_q_values(gamma=0.9)
            q_data = collect_q_data_from_a_tree(post_processed_t)
            if upper_num is not None and len(q_data) > upper_num:
                q_data = random.sample(q_data, upper_num)
                
            q_data = normalize_data(q_data, 'label')
            q_data = [{'id': id, **entry} for entry in q_data]
            q_data_ls.extend(q_data)

    with open(save_q_pth, 'w') as f:
        json.dump(q_data_ls, f, indent=4)
        
    print(f'saved {len(q_data_ls)} data points')

def collect_r_data_from_trees_unpruned(tree_pth,save_r_pth):
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    r_data_ls = []
    for tree in tqdm(trees[0]):
        for t in tree:
            id = t['id']
            # post_processed_t = post_process_tree(t['tree'])
            post_processed_t = t['tree']
            post_processed_t.update_rewards()
            r_data = collect_r_data_from_a_tree(post_processed_t)
            
            r_data = normalize_data(r_data, 'label')
            r_data = [{'id': id, **entry} for entry in r_data]
            r_data_ls.extend(r_data)
    with open(save_r_pth, 'w') as f:
        json.dump(r_data_ls, f, indent=4)

def main(args):
    
    jsonl_files = [os.path.join(args.data_path,data_name) for data_name in os.listdir(args.data_path) if data_name.endswith('.jsonl') and 'combined' not in data_name]
    pkl_files = [os.path.join(args.data_path,data_name) for data_name in os.listdir(args.data_path) if data_name.endswith('.pkl') and 'combined' not in data_name]
    combined_traj_file = os.path.join(args.data_path.replace('/projects/prjs1381/', '/scratch-shared/yzhang2/'), 'combined_traj.jsonl')
    combined_tree_file = os.path.join(args.data_path.replace('/projects/prjs1381/', '/scratch-shared/yzhang2/'), 'combined_tree.pkl')
    q_file = os.path.join(args.data_path, f'q_data-expectile{args.expectile}.jsonl')
    r_file = os.path.join(args.data_path, 'r_data.jsonl')
    # Combined slices
    combine_jsonl(jsonl_files, combined_traj_file)
    combine_pkl(pkl_files, combined_tree_file)
    q_file = 'data/train/'+args.task+'/explore/'+f'{args.q_type}.jsonl'
    
    if args.q_type == 'vanilla':
        collect_vanilla_q_data_from_trees_unpruned(combined_tree_file, q_file)
    elif args.q_type == 'pseudo_depth':
        collect_q_data_from_trees_unpruned(combined_tree_file, q_file) ## pseudo children nodes
    elif args.q_type == 'reward':
        collect_r_data_from_trees_unpruned(combined_tree_file, q_file)
    else:
        raise ValueError(f"q_type {args.q_type} is not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--task', type=str, default='webshop', help='task name')
    parser.add_argument('--data_path', type=str, default='data/train/webshop/self_explore_iter2/',help='data path')
    parser.add_argument('--q_type', type=str, default='vanilla', help='q_type')
    parser.add_argument('--expectile', type=float, default=0.7, help='expectile')
    args = parser.parse_args()
    main(args)