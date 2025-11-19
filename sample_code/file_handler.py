"""
File handler with resource management issues
"""

import os

def read_file(filename):
    """Read file - resource leak"""
    # Bug: File not closed properly
    file = open(filename, 'r')
    data = file.read()
    return data

def write_data(filename, data):
    """Write data - no error handling"""
    # Bug: No exception handling
    file = open(filename, 'w')
    file.write(data)
    file.close()

def process_files(directory):
    """Process multiple files - inefficient"""
    results = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Bug: Path concatenation issues
            full_path = directory + '/' + filename

            try:
                data = read_file(full_path)
                results.append(data)
            except:
                # Bug: Silent failure
                pass

    return results

def copy_file(source, destination):
    """Copy file - no validation"""
    # Bug: No existence check
    data = read_file(source)
    write_data(destination, data)

def delete_files(pattern, directory):
    """Delete files matching pattern"""
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                # Bug: No confirmation, dangerous
                file_path = os.path.join(root, file)
                os.remove(file_path)
                count += 1

    return count
