#!/usr/bin/env python3
'''Quick test script to verify Session 7 installation'''

import sys
import os

def test_imports():
    '''Test that all modules can be imported'''
    print("Testing Session 7 imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
        
        from session7_uvvisnir_analyzer import UVVisNIRAnalyzer
        print("✓ UV-Vis-NIR analyzer imported successfully")
        
        from session7_ftir_analyzer import FTIRAnalyzer
        print("✓ FTIR analyzer imported successfully")
        
        # Quick functionality test
        uv = UVVisNIRAnalyzer()
        ftir = FTIRAnalyzer()
        print("✓ Analyzers instantiated successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Session 7 modules are ready to use!")
    else:
        print("\n❌ Please check installation and dependencies")
    sys.exit(0 if success else 1)
