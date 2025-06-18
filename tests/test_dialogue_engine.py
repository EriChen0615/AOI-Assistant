#!/usr/bin/env python3
"""
Test file for AOI Dialogue Engine
Tests the _get_model_response method and function calling capabilities
"""

import sys
import os
import json
import tempfile

# Add src to path
sys.path.append('src')

from aoi.dialogue_engine import AOIDialogueEngine, get_model_response

def test_basic_response():
    """Test basic text response without function calling"""
    print("=== Testing Basic Response ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary API key file with real API key
        api_key_file = os.path.join(temp_dir, "api_key")
        with open("configs/openai_api_key", 'r') as f:
            api_key = f.read().strip()
        with open(api_key_file, 'w') as f:
            f.write(api_key)
        
        try:
            # Initialize dialogue engine
            engine = AOIDialogueEngine(
                model_name="gpt-4o-mini",
                save_dir=temp_dir,
                api_key_file=api_key_file
            )
            
            # Test basic response
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
            
            print("Testing basic response...")
            response = engine._get_model_response(
                messages=messages,
                model_name_or_path="gpt-4o-mini",
                max_tokens=50,
                n_seqs=1,
                temperature=0.0
            )
            
            print(f"Response: {response}")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            if response and len(response) > 0:
                print(f"First candidate: {response[0]}")
                print(f"Content: {response[0].get('content', 'No content')}")
                print("✓ Basic response test passed")
            else:
                print("✗ Basic response test failed - no response")
                
        except Exception as e:
            print(f"✗ Basic response test failed with error: {e}")
            import traceback
            traceback.print_exc()

def test_function_calling():
    """Test function calling capabilities"""
    print("\n=== Testing Function Calling ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary API key file with real API key
        api_key_file = os.path.join(temp_dir, "api_key")
        with open("configs/openai_api_key", 'r') as f:
            api_key = f.read().strip()
        with open(api_key_file, 'w') as f:
            f.write(api_key)
        
        try:
            # Initialize dialogue engine
            engine = AOIDialogueEngine(
                model_name="gpt-4o-mini",
                save_dir=temp_dir,
                api_key_file=api_key_file
            )
            
            # Test function calling
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can search for information."},
                {"role": "user", "content": "Search for information about Python programming"}
            ]
            
            # Define a test function
            test_function = {
                "type": "function",
                "function": {
                    "name": "search_wiki",
                    "description": "Search wikipedia for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search wikipedia for"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            
            print("Testing function calling...")
            response = engine._get_model_response(
                messages=messages,
                model_name_or_path="gpt-4o-mini",
                max_tokens=100,
                n_seqs=1,
                functions=[test_function],
                function_call={"name": "search_wiki"},
                temperature=0.0
            )
            
            print(f"Response: {response}")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            if response and len(response) > 0:
                first_response = response[0]
                print(f"First candidate: {first_response}")
                print(f"Content: {first_response.get('content', 'No content')}")
                print(f"Function call: {first_response.get('function_call', 'No function call')}")
                
                if first_response.get('function_call'):
                    print("✓ Function calling test passed")
                else:
                    print("⚠ Function calling test - no function call returned (this might be expected)")
            else:
                print("✗ Function calling test failed - no response")
                
        except Exception as e:
            print(f"✗ Function calling test failed with error: {e}")
            import traceback
            traceback.print_exc()

def test_standalone_get_model_response():
    """Test the standalone get_model_response function"""
    print("\n=== Testing Standalone get_model_response ===")
    
    try:
        # Test basic response
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ]
        
        print("Testing standalone get_model_response...")
        response = get_model_response(
            messages=messages,
            model_name_or_path="gpt-4o-mini",
            max_tokens=30,
            n_seqs=1,
            temperature=0.0
        )
        
        print(f"Response: {response}")
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response)}")
        
        if response and len(response) > 0:
            print(f"First candidate: {response[0]}")
            print(f"Content: {response[0].get('content', 'No content')}")
            print("✓ Standalone get_model_response test passed")
        else:
            print("✗ Standalone get_model_response test failed - no response")
            
    except Exception as e:
        print(f"✗ Standalone get_model_response test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_dialogue_engine_run_turn():
    """Test the complete run_turn method"""
    print("\n=== Testing Dialogue Engine run_turn ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary API key file with real API key
        api_key_file = os.path.join(temp_dir, "api_key")
        with open("configs/openai_api_key", 'r') as f:
            api_key = f.read().strip()
        with open(api_key_file, 'w') as f:
            f.write(api_key)
        
        try:
            # Initialize dialogue engine
            engine = AOIDialogueEngine(
                model_name="gpt-4o-mini",
                save_dir=temp_dir,
                api_key_file=api_key_file
            )
            
            print("Testing run_turn with simple query...")
            response = engine.run_turn("Hello, how are you?")
            
            print(f"Response: {response}")
            print(f"Response type: {type(response)}")
            
            if response:
                print("✓ run_turn test passed")
            else:
                print("✗ run_turn test failed - no response")
                
        except Exception as e:
            print(f"✗ run_turn test failed with error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run all tests"""
    print("Starting AOI Dialogue Engine Tests")
    print("=" * 50)
    
    # Check if API key is available
    api_key_file = "configs/openai_api_key"
    if os.path.exists(api_key_file):
        print(f"✓ Found API key file: {api_key_file}")
        
        # Run tests with real API key
        test_basic_response()
        test_function_calling()
        test_standalone_get_model_response()
        test_dialogue_engine_run_turn()
    else:
        print(f"✗ API key file not found: {api_key_file}")
        print("Please create the API key file or update the test to use your API key")
        print("Tests will be skipped.")

if __name__ == "__main__":
    main() 