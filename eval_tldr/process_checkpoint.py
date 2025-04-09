import os
import shutil
from safetensors import safe_open
from safetensors.torch import save_file, safe_open
import json
import argparse

def process_checkpoint(checkpoint_path):
    """
    Copies a checkpoint folder and modifies its safetensors file to remove 'module.' from keys.
    
    Args:
        checkpoint_path (str): Path to the checkpoint folder
    """
    # Create the new folder name by appending '-no-module'
    base_path = os.path.dirname(checkpoint_path.rstrip('/'))
    checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
    new_folder = os.path.join(base_path, f"{checkpoint_name}-NM")
    
    # Copy the entire folder
    print(f"Copying {checkpoint_path} to {new_folder}")
    shutil.copytree(checkpoint_path, new_folder, dirs_exist_ok=True)
    
    # Find and process safetensors files
    for file in os.listdir(new_folder):
        if file.endswith('.safetensors'):
            safetensors_path = os.path.join(new_folder, file)
            print(f"Processing {safetensors_path}")
            
            # Load the original tensors and metadata
            tensors = {}
            metadata = None
            with safe_open(safetensors_path, framework="pt") as f:
                # Get metadata
                metadata = f.metadata()
                if metadata is None:
                    metadata = {"format": "pt"}  # Add default metadata
                elif isinstance(metadata, bytes):
                    metadata = json.loads(metadata.decode('utf-8'))
                
                # Get tensors
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            
            # Create new dict with modified keys
            new_tensors = {}
            for key, tensor in tensors.items():
                new_key = key.replace('module.', '')
                new_tensors[new_key] = tensor
            
            # Save the modified tensors with metadata
            print(f"Saving modified tensors to {safetensors_path}")
            save_file(new_tensors, safetensors_path, metadata=metadata)
            
            print(f"Successfully processed {file}")
        elif file.endswith('.safetensors.index.json'):
            json_path = os.path.join(new_folder, file)
            print(f"Processing {json_path}")

            # Read the existing file
            with open(json_path, 'r') as f:
                orig = json.load(f)

            # Process the weight map
            remove = set()
            weight_map = orig['weight_map']
            for key in list(weight_map.keys()):
                if 'module.' in key:
                    remove.add(key)
                    new_key = key.replace('module.', '')
                    weight_map[new_key] = weight_map[key]

            # Remove old keys
            for key in remove:
                del weight_map[key]

            # Save back to file
            with open(json_path, 'w') as f:
                json.dump(orig, f)

            print(f"Successfully processed {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a checkpoint to remove module. prefix from keys.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint folder')
    args = parser.parse_args()
    
    process_checkpoint(args.checkpoint_path)
