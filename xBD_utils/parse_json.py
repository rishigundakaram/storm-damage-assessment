import json
import os
from shapely.wkt import loads
import argparse

def convert_polygon_to_bounds_and_label_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')

            with open(input_file_path, 'r') as file:
                data = json.load(file)

            buildings_info = []

            for feature in data['features']['xy']:  # Changed from 'lng_lat' to 'xy'
                properties = feature['properties']
                polygon = loads(feature['wkt'])
                bounds = polygon.bounds  # Gets the bounding box as (minx, miny, maxx, maxy)
                print(properties)
                # Assign label based on subtype
                try:
                    assert properties['feature_type'] == 'building'
                    label = 0 if properties['subtype'] == 'no-damage' else 1

                    buildings_info.append({
                        'bounds': bounds,
                        'label': label
                    })
                except KeyError:
                    print(f"Skipping building with UID {properties['uid']} because it has no subtype")

            # Optionally, save the processed data to a new JSON file in the output directory
            with open(output_file_path, 'w') as outfile:
                json.dump(buildings_info, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert polygon to bounds and label for object detection.")
    parser.add_argument("input_dir", help="Input directory containing JSON files")
    parser.add_argument("output_dir", help="Output directory for processed JSON files")
    
    args = parser.parse_args()
    
    convert_polygon_to_bounds_and_label_directory(args.input_dir, args.output_dir)
