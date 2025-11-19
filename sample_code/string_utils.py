"""
String utility functions - Moderate complexity
"""

def reverse_string(s):
    """Reverse a string"""
    return s[::-1]

def count_vowels(text):
    """Count vowels in text"""
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

def remove_duplicates(text):
    """Remove duplicate characters"""
    seen = set()
    result = []
    for char in text:
        if char not in seen:
            seen.add(char)
            result.append(char)
    return ''.join(result)

def is_palindrome(text):
    """Check if text is palindrome"""
    # Bug: Case sensitive
    clean = text.replace(' ', '')
    return clean == clean[::-1]

def word_frequency(text):
    """Calculate word frequency"""
    words = text.split()
    freq = {}
    for word in words:
        # Bug: Case sensitive counting
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq
