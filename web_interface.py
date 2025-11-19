#!/usr/bin/env python3
"""
Web Interface for Unified Defect Mitigation Framework
Allows users to input code and receive defect predictions, localizations, and fixes
"""

import streamlit as st
import sys
from main_framework import UnifiedDefectMitigationFramework
import json

# Page configuration
st.set_page_config(
    page_title="Defect Mitigation Framework",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Unified Defect Mitigation Framework")
st.markdown("""
This system provides end-to-end software defect mitigation through three integrated phases:
- **Phase 1**: ML-based Defect Prediction
- **Phase 2**: GAT-based Defect Localization
- **Phase 3**: RATG-based Bug Fix Generation
""")

# Initialize framework (with caching)
@st.cache_resource
def load_framework():
    return UnifiedDefectMitigationFramework()

try:
    framework = load_framework()
    st.success("✓ Framework loaded successfully!")
except Exception as e:
    st.error(f"Error loading framework: {e}")
    st.stop()

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
show_phase1 = st.sidebar.checkbox("Phase 1: Defect Prediction", value=True)
show_phase2 = st.sidebar.checkbox("Phase 2: Defect Localization", value=True)
show_phase3 = st.sidebar.checkbox("Phase 3: Bug Fix Generation", value=True)

# Main input area
st.header("📝 Input Code")

# Sample code for testing
sample_code = '''def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    avg = total / len(numbers)  # Potential division by zero
    return avg

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result[0]  # Potential IndexError
'''

code_input = st.text_area(
    "Enter your Python code here:",
    value=sample_code,
    height=300,
    help="Enter the Python code you want to analyze"
)

file_name = st.text_input("File name (optional)", value="input_code.py")

# Analyze button
if st.button("🔍 Analyze Code", type="primary"):
    if not code_input.strip():
        st.error("Please enter some code to analyze!")
    else:
        with st.spinner("Analyzing code..."):
            # Process through framework
            test_files = [{'path': file_name, 'code': code_input}]

            try:
                results = framework.process(test_files)

                if not results:
                    st.error("No results returned from framework")
                else:
                    result = results[0]  # Get first result

                    # Display results in tabs
                    tab1, tab2, tab3, tab_summary = st.tabs([
                        "📊 Phase 1: Prediction",
                        "🎯 Phase 2: Localization",
                        "🔧 Phase 3: Fix Generation",
                        "📋 Summary"
                    ])

                    with tab1:
                        if show_phase1:
                            st.subheader("Phase 1: Defect Prediction Results")

                            phase1 = result.get('phase1', {})
                            is_defective = phase1.get('is_defective', False)
                            prob = phase1.get('defect_probability', 0.0)

                            col1, col2 = st.columns(2)

                            with col1:
                                if is_defective:
                                    st.error(f"⚠️ **DEFECTIVE** ({prob*100:.1f}% probability)")
                                else:
                                    st.success(f"✓ **CLEAN** ({(1-prob)*100:.1f}% confidence)")

                            with col2:
                                st.metric("Defect Probability", f"{prob:.2%}")

                            # Progress bar
                            st.progress(prob)

                            # Metrics display
                            st.markdown("---")
                            st.markdown("**Evaluation Metrics:**")
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                            with metrics_col1:
                                st.metric("Target Accuracy", "≥85%")
                            with metrics_col2:
                                st.metric("Target F1-Score", "≥0.85")
                            with metrics_col3:
                                st.metric("Target ROC-AUC", "≥0.85")
                        else:
                            st.info("Phase 1 is disabled")

                    with tab2:
                        if show_phase2 and result.get('phase2'):
                            st.subheader("Phase 2: Defect Localization Results")

                            phase2 = result.get('phase2', {})
                            suspicious_lines = phase2.get('suspicious_lines', [])

                            if suspicious_lines:
                                st.warning(f"Found {len(suspicious_lines)} suspicious locations")

                                st.markdown("**Top Suspicious Lines:**")
                                for i, line in enumerate(suspicious_lines, 1):
                                    st.markdown(f"{i}. Line {line}")

                                # Show code with highlighted lines
                                st.markdown("---")
                                st.markdown("**Code with Suspicious Lines Highlighted:**")

                                code_lines = code_input.split('\n')
                                highlighted_code = []
                                for i, line in enumerate(code_lines, 1):
                                    if i in suspicious_lines:
                                        highlighted_code.append(f"⚠️ {i:3d} | {line}")
                                    else:
                                        highlighted_code.append(f"   {i:3d} | {line}")

                                st.code('\n'.join(highlighted_code), language='python')
                            else:
                                st.info("No specific suspicious lines identified")
                        else:
                            st.info("Phase 2 is disabled or no defects detected")

                    with tab3:
                        if show_phase3 and result.get('phase3'):
                            st.subheader("Phase 3: Bug Fix Generation Results")

                            phase3 = result.get('phase3', {})
                            fixed_code = phase3.get('fixed_code', '')
                            applied_fixes = phase3.get('applied_fixes', [])

                            if applied_fixes:
                                st.success(f"✓ Applied {len(applied_fixes)} fix(es)")

                                # Show applied fixes
                                st.markdown("**Applied Fixes:**")
                                for i, fix in enumerate(applied_fixes, 1):
                                    with st.expander(f"Fix {i}: {fix.get('fix_type', 'Unknown')} (confidence: {fix.get('confidence', 0):.2f})"):
                                        st.markdown(f"**Description:** {fix.get('description', 'N/A')}")
                                        st.markdown(f"**Line:** {fix.get('line', 'N/A')}")

                                # Show fixed code
                                st.markdown("---")
                                st.markdown("**Fixed Code:**")
                                st.code(fixed_code, language='python')

                                # Download button
                                st.download_button(
                                    label="📥 Download Fixed Code",
                                    data=fixed_code,
                                    file_name=f"fixed_{file_name}",
                                    mime="text/plain"
                                )
                            else:
                                st.info("No fixes were applied. Code may already be correct or no patterns matched.")
                        else:
                            st.info("Phase 3 is disabled or no defects to fix")

                    with tab_summary:
                        st.subheader("📋 Complete Analysis Summary")

                        # Overall status
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            phase1 = result.get('phase1', {})
                            status1 = "✓" if not phase1.get('is_defective', False) else "⚠️"
                            st.metric("Phase 1", status1, "Prediction")

                        with col2:
                            phase2 = result.get('phase2', {})
                            lines = len(phase2.get('suspicious_lines', [])) if phase2 else 0
                            st.metric("Phase 2", lines, "Suspicious Lines")

                        with col3:
                            phase3 = result.get('phase3', {})
                            fixes = len(phase3.get('applied_fixes', [])) if phase3 else 0
                            st.metric("Phase 3", fixes, "Fixes Applied")

                        # JSON results
                        st.markdown("---")
                        st.markdown("**Detailed Results (JSON):**")
                        st.json(result)

                        # Download JSON
                        st.download_button(
                            label="📥 Download Full Results (JSON)",
                            data=json.dumps(result, indent=2),
                            file_name="analysis_results.json",
                            mime="application/json"
                        )

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
### About This Framework

This framework implements a three-phase approach to software defect mitigation:

1. **Phase 1 - Defect Prediction (ML-based)**:
   - Ensemble model: Random Forest + SVM + Decision Tree
   - SMOTE-TOMEK for class balancing
   - Mutual Information feature selection
   - Target: F1≥85%, AUC≥0.85, Accuracy≥85%

2. **Phase 2 - Defect Localization (GAT-based)**:
   - Graph Attention Network on AST
   - 2-layer GAT with multi-head attention
   - 18,177 trainable parameters
   - Target: Top-3 localization accuracy ≥70%

3. **Phase 3 - Bug Fix Generation (RATG)**:
   - Retrieval-Augmented Template Generation
   - CodeBERT-based retrieval (concept)
   - Template database with common bug patterns
   - Target: Valid fix rate ≥80%

**Trained on NASA Promise Datasets**: CM1, JM1, KC1, KC2, PC1 (15,123 total samples)
""")

st.markdown("---")
st.caption("© 2025 Unified Defect Mitigation Framework | Powered by Machine Learning & Graph Neural Networks")
