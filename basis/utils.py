# utility functions

def is_mesh_empty(mesh):
    return len(mesh.vertices) == 0 and len(mesh.faces) == 0

def get_yesno():
    while True:
        user_input = input("Please enter 'y' or 'n': ").lower()  # Convert input to lowercase for case-insensitivity
        if user_input == 'y' or user_input == 'n':
            return user_input
        else:
            print("Invalid input. Must be 'y' or 'n', other input not acceptable.")