import os
import json
from tqdm import tqdm

def generate_shapenet_json(shapenet_dir, output_json, target_class):
    """
    Generate a JSON file mapping ShapeNet class acronyms to model UUIDs.
    
    Args:
        shapenet_dir (str): Path to ShapeNet dataset (e.g., "ShapeNetCore.v2/").
        output_json (str): Path to save the JSON file (e.g., "shapenet_mugs.json").
        target_class (str): Class name (e.g., "Mug", "Couch", "Chair").
    """
    # ShapeNet class to acronym mapping (adjust if needed)
    class_to_acronym = {
        "Mug": "03797390",
        "Couch": "04256520",
        "Chair": "03001627",
        "Table": "04379243",
        # Add more classes as needed (refer to ShapeNet's taxonomy.json)
    }

    acronym = class_to_acronym.get(target_class)
    if not acronym:
        raise ValueError(f"Class '{target_class}' not found in ShapeNet's acronym mapping.")

    # Scan the ShapeNet directory for models of the target class
    class_dir = os.path.join(shapenet_dir, acronym)
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Directory for class '{target_class}' ({acronym}) not found in {shapenet_dir}, {class_dir}.")

    model_ids = [
        model_id for model_id in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, model_id))
    ]

    # Generate the JSON structure
    data = {
        "acronym": {
            target_class: model_ids
        }
    }

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {len(model_ids)} {target_class} models to {output_json}.")

if __name__ == "__main__":
    # Example usage
    shapenet_dir = "/home/iacopo/Diffusion-SDF/shapenet_download/ShapeNetCore.v2"
    output_json = "mug_all.json"         # Output filename
    target_class = "Mug"                       # Change to "Couch", "Chair", etc.

    generate_shapenet_json(shapenet_dir, output_json, target_class)