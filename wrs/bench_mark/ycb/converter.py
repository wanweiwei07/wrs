import os
import wrs.modeling.mesh_tools as mt

if __name__ == '__main__':
    current_folder = os.getcwd()  # Get current working directory
    ply_files = [f for f in os.listdir(current_folder) if f.endswith('.ply')]

    # Loop through each .ply file and convert it to .stl
    for ply_file in ply_files:
        # Define the output .stl file name
        stl_file = ply_file.replace('.ply', '.stl')

        # Call the convert_to_stl function
        mt.convert_to_stl(ply_file, stl_file)