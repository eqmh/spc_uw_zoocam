import os

def count_png_files_in_subdirs(main_directory):
    """
    Counts the number of .png files in each immediate subdirectory.

    Args:
        main_directory (str): The path to the main directory.

    Returns:
        list: A list of tuples, where each tuple contains the
              subdirectory name and the count of .png files.
    """
    # Check if the main directory exists
    if not os.path.isdir(main_directory):
        print(f"Error: Directory not found at '{main_directory}'")
        return []

    counts_list = []
    # Iterate through items in the main directory
    for subdir_name in sorted(os.listdir(main_directory)):
        subdir_path = os.path.join(main_directory, subdir_name)
        
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            png_count = 0
            # Iterate through files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith('.png'):
                    png_count += 1
            
            counts_list.append((subdir_name, png_count))
            
    return counts_list

# --- Main execution part ---
if __name__ == "__main__":
    # Set the path to your 'training_library' directory
    training_library_path = '/Users/enrique.montes/Desktop/uw_classifier/augmented_categories/' # Or use a full path like '/path/to/training_library'
    
    # Get the list of counts
    file_counts = count_png_files_in_subdirs(training_library_path)
    
    # Print the results
    if file_counts:
        print(f"PNG file counts in the subdirectories of '{training_library_path}':")
        for folder, count in file_counts:
            print(f"- {folder}: {count} files")