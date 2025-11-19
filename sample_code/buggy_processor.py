"""
Data processor with multiple defects
"""

def process_data(data):
    """Process a list of data items - HAS BUGS"""
    results = []

    # Bug: Index out of range
    for i in range(len(data)):
        value = data[i + 1]  # Off-by-one error
        results.append(value * 2)

    return results

def calculate_average(numbers):
    """Calculate average - HAS BUGS"""
    # Bug: Division by zero not handled
    total = sum(numbers)
    return total / len(numbers)

def find_max(items):
    """Find maximum value - HAS BUGS"""
    max_val = 0  # Bug: Assumes all positive numbers
    for item in items:
        if item > max_val:
            max_val = item
    return max_val

def merge_dicts(dict1, dict2):
    """Merge two dictionaries - HAS BUGS"""
    result = dict1
    # Bug: Modifies original dict
    for key, value in dict2.items():
        result[key] = value
    return result

def parse_config(config_string):
    """Parse configuration - HAS BUGS"""
    try:
        # Bug: Bare except catches everything
        data = eval(config_string)  # Security vulnerability
        return data
    except:
        pass
    return None

def complex_calculation(x, y, z):
    """Perform complex calculation with high complexity"""
    result = 0

    if x > 0:
        if y > 0:
            if z > 0:
                for i in range(x):
                    for j in range(y):
                        for k in range(z):
                            if i == j:
                                if j == k:
                                    result += i * j * k
                                else:
                                    result += i + j + k
                            else:
                                if k % 2 == 0:
                                    result += i - j + k
                                else:
                                    result += i + j - k
            else:
                result = x + y
        else:
            result = x
    else:
        result = 0

    return result

class DataManager:
    """Data manager with defects"""

    def __init__(self):
        self.data = []

    def add_item(self, item):
        """Add item without validation"""
        self.data.append(item)

    def get_item(self, index):
        """Get item - no bounds checking"""
        return self.data[index]

    def remove_item(self, item):
        """Remove item - doesn't check existence"""
        self.data.remove(item)

    def process_all(self):
        """Process all items - inefficient"""
        results = []
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if i != j:
                    results.append(self.data[i] + self.data[j])
        return results
