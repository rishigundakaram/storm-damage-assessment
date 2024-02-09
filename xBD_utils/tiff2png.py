import os
import cv2
from PIL import Image
from osgeo import gdal

def convert_tiff_to_jpeg(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            # Open the geospatial TIFF file with GDAL
            ds = gdal.Open(input_path)
            if ds is None:
                print(f"Failed to open {input_path}")
                continue
            
            # Convert to JPEG using GDAL Translate function
            gdal.Translate(output_path, ds, format='JPEG', outputType=gdal.GDT_Byte, scaleParams=[])

            print(f"Converted {filename} to JPEG")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert tiff images to jpeg')
    parser.add_argument('input_dir', type=str, help='Directory containing tiff images')
    parser.add_argument('output_dir', type=str, help='Output directory for jpeg images')
    args = parser.parse_args()

    convert_tiff_to_jpeg(args.input_dir, args.output_dir)