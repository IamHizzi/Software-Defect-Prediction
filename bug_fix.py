"""
Bug Fix Module
Phase 3: Retrieval-Augmented Template Generation (RATG)
"""

import ast
import difflib
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class BugFixTemplate:
    """Represents a bug fix template"""
    
    def __init__(self, bug_pattern, fix_pattern, context=None, bug_type=None):
        self.bug_pattern = bug_pattern
        self.fix_pattern = fix_pattern
        self.context = context or {}
        self.bug_type = bug_type
    
    def __repr__(self):
        return f"Template(type={self.bug_type}, pattern={self.bug_pattern[:50]}...)"


class BugFixRepository:
    """Repository of historical bug fixes"""
    
    def __init__(self):
        self.fixes = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.fix_vectors = None
        self._initialize_common_fixes()
    
    def _initialize_common_fixes(self):
        """Initialize with common bug fix patterns"""
        common_fixes = [
            {
                'bug': 'x / 0',
                'fix': 'x / max(y, 1)',
                'bug_type': 'division_by_zero',
                'context': {'pattern': r'(\w+)\s*/\s*0', 'replacement': r'\1 / max(\1, 1)'}
            },
            {
                'bug': 'items[i+1]',
                'fix': 'items[min(i+1, len(items)-1)]',
                'bug_type': 'index_out_of_bounds',
                'context': {'pattern': r'(\w+)\[(\w+)\+1\]', 'replacement': r'\1[min(\2+1, len(\1)-1)]'}
            },
            {
                'bug': 'if x == None:',
                'fix': 'if x is None:',
                'bug_type': 'none_comparison',
                'context': {'pattern': r'(\w+)\s*==\s*None', 'replacement': r'\1 is None'}
            },
            {
                'bug': 'except:',
                'fix': 'except Exception as e:',
                'bug_type': 'bare_except',
                'context': {'pattern': r'except\s*:', 'replacement': 'except Exception as e:'}
            },
            {
                'bug': 'while True:',
                'fix': 'while condition:',
                'bug_type': 'infinite_loop',
                'context': {'pattern': r'while\s+True\s*:', 'replacement': 'while <condition>:'}
            },
            {
                'bug': 'return x / len(items)',
                'fix': 'return x / len(items) if items else 0',
                'bug_type': 'empty_list_division',
                'context': {'pattern': r'(\w+)\s*/\s*len\((\w+)\)', 
                           'replacement': r'\1 / len(\2) if \2 else 0'}
            },
            {
                'bug': 'dict[key]',
                'fix': 'dict.get(key, default)',
                'bug_type': 'key_error',
                'context': {'pattern': r'(\w+)\[(["\']?\w+["\']?)\]', 
                           'replacement': r'\1.get(\2, None)'}
            },
            {
                'bug': 'file = open(path)',
                'fix': 'with open(path) as file:',
                'bug_type': 'unclosed_file',
                'context': {'pattern': r'(\w+)\s*=\s*open\((.*?)\)', 
                           'replacement': r'with open(\2) as \1:'}
            }
        ]
        
        for fix_data in common_fixes:
            template = BugFixTemplate(
                bug_pattern=fix_data['bug'],
                fix_pattern=fix_data['fix'],
                context=fix_data.get('context', {}),
                bug_type=fix_data.get('bug_type')
            )
            self.fixes.append(template)
    
    def add_fix(self, bug_code, fixed_code, bug_type=None):
        """Add a new fix to the repository"""
        # Extract difference pattern
        context = self._extract_diff_pattern(bug_code, fixed_code)
        
        template = BugFixTemplate(
            bug_pattern=bug_code,
            fix_pattern=fixed_code,
            context=context,
            bug_type=bug_type
        )
        self.fixes.append(template)
    
    def _extract_diff_pattern(self, bug_code, fixed_code):
        """Extract pattern from bug and fix"""
        # Simple diff-based pattern extraction
        differ = difflib.Differ()
        diff = list(differ.compare(bug_code.split(), fixed_code.split()))
        
        removed = [item[2:] for item in diff if item.startswith('- ')]
        added = [item[2:] for item in diff if item.startswith('+ ')]
        
        return {
            'removed': ' '.join(removed),
            'added': ' '.join(added)
        }
    
    def build_index(self):
        """Build search index for retrieval"""
        bug_texts = [fix.bug_pattern for fix in self.fixes]
        if bug_texts:
            self.fix_vectors = self.vectorizer.fit_transform(bug_texts)
    
    def retrieve_similar_fixes(self, buggy_code, top_k=3):
        """Retrieve most similar fixes from repository"""
        if self.fix_vectors is None:
            self.build_index()
        
        # Vectorize query
        query_vector = self.vectorizer.transform([buggy_code])
        
        # Compute similarity
        similarities = cosine_similarity(query_vector, self.fix_vectors).flatten()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_fixes = [(self.fixes[idx], similarities[idx]) for idx in top_indices]
        return similar_fixes


class BugFixer:
    """
    Retrieval-Augmented Template Generation for bug fixing
    """
    
    def __init__(self):
        self.repository = BugFixRepository()
        self.repository.build_index()
    
    def generate_fix(self, buggy_code, buggy_lines=None, top_k=3):
        """
        Generate fix for buggy code
        
        Args:
            buggy_code: The buggy source code
            buggy_lines: List of line numbers identified as buggy
            top_k: Number of similar fixes to retrieve
            
        Returns:
            Fixed code with applied patches
        """
        # Retrieve similar fixes
        similar_fixes = self.repository.retrieve_similar_fixes(buggy_code, top_k)
        
        if not similar_fixes:
            return buggy_code, []
        
        # Try to apply fixes
        fixed_code = buggy_code
        applied_fixes = []
        
        for template, similarity in similar_fixes:
            if similarity < 0.1:  # Skip very dissimilar fixes
                continue
            
            # Try pattern-based fix
            if 'pattern' in template.context and 'replacement' in template.context:
                pattern = template.context['pattern']
                replacement = template.context['replacement']
                
                try:
                    new_code = re.sub(pattern, replacement, fixed_code)
                    if new_code != fixed_code:
                        fixed_code = new_code
                        applied_fixes.append({
                            'template': template,
                            'similarity': similarity,
                            'type': template.bug_type
                        })
                except:
                    pass
        
        # If no pattern match, try line-based replacement
        if not applied_fixes and buggy_lines:
            fixed_code = self._apply_line_fixes(buggy_code, buggy_lines, similar_fixes)
            if fixed_code != buggy_code:
                applied_fixes.append({
                    'template': similar_fixes[0][0],
                    'similarity': similar_fixes[0][1],
                    'type': 'line_replacement'
                })
        
        return fixed_code, applied_fixes
    
    def _apply_line_fixes(self, code, buggy_lines, similar_fixes):
        """Apply fixes to specific lines"""
        lines = code.split('\n')
        
        for line_num in buggy_lines:
            if 0 < line_num <= len(lines):
                idx = line_num - 1
                original_line = lines[idx]
                
                # Try to apply most similar fix
                for template, _ in similar_fixes[:1]:
                    # Simple heuristic: add defensive checks
                    if 'len(' in original_line and '/' in original_line:
                        # Add empty check for division
                        lines[idx] = original_line.replace(
                            'return', 
                            'return 0 if not items else'
                        )
                    elif '[' in original_line and '+' in original_line:
                        # Add bounds check
                        var_match = re.search(r'(\w+)\[(\w+)\+\d+\]', original_line)
                        if var_match:
                            var, idx_var = var_match.groups()
                            lines[idx] = re.sub(
                                rf'{var}\[{idx_var}\+\d+\]',
                                f'{var}[min({idx_var}+1, len({var})-1)]',
                                original_line
                            )
                    break
        
        return '\n'.join(lines)
    
    def generate_multiple_fixes(self, buggy_code, n_candidates=3):
        """Generate multiple fix candidates"""
        candidates = []
        
        similar_fixes = self.repository.retrieve_similar_fixes(buggy_code, top_k=5)
        
        for i, (template, similarity) in enumerate(similar_fixes[:n_candidates]):
            fixed_code, applied = self.generate_fix(buggy_code, top_k=1)
            candidates.append({
                'rank': i + 1,
                'fixed_code': fixed_code,
                'template': template,
                'similarity': similarity,
                'applied_fixes': applied
            })
        
        return candidates
    
    def explain_fix(self, applied_fixes):
        """Generate explanation for applied fixes"""
        explanations = []
        
        for fix in applied_fixes:
            template = fix['template']
            explanation = f"Applied fix for {template.bug_type or 'unknown bug'}:\n"
            explanation += f"  Pattern: {template.bug_pattern}\n"
            explanation += f"  Fix: {template.fix_pattern}\n"
            explanation += f"  Confidence: {fix['similarity']:.2f}\n"
            explanations.append(explanation)
        
        return '\n'.join(explanations)


def demo_bug_fixing():
    """Demonstrate bug fixing"""
    
    buggy_codes = [
        # Example 1: Division by zero
        """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
""",
        # Example 2: Index out of bounds
        """
def get_next_items(items):
    result = []
    for i in range(len(items)):
        result.append(items[i+1])
    return result
""",
        # Example 3: None comparison
        """
def check_value(x):
    if x == None:
        return False
    return True
""",
        # Example 4: Bare except
        """
def safe_divide(a, b):
    try:
        return a / b
    except:
        return None
"""
    ]
    
    fixer = BugFixer()
    
    for i, buggy_code in enumerate(buggy_codes, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: Bug Fix")
        print('='*60)
        print("\nOriginal (Buggy) Code:")
        print(buggy_code)
        
        # Generate fix
        fixed_code, applied_fixes = fixer.generate_fix(buggy_code)
        
        print("\nFixed Code:")
        print(fixed_code)
        
        if applied_fixes:
            print("\nExplanation:")
            print(fixer.explain_fix(applied_fixes))
        else:
            print("\nNo automatic fix applied (may need manual review)")


if __name__ == "__main__":
    print("Bug Fix Module - Retrieval-Augmented Template Generation")
    demo_bug_fixing()