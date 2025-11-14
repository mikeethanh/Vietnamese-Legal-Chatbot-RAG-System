#!/usr/bin/env python3
"""
Test script to verify WANDB API key and entity permissions
for the vietnamese-legal-llama-unsloth project
"""

import os
import wandb
from dotenv import load_dotenv

def test_wandb_permissions():
    """Test WANDB API key and entity permissions"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv('WANDB_API_KEY')
    entity = os.getenv('WANDB_ENTITY')
    project_name = "vietnamese-legal-llama-unsloth"
    
    print(f"🔑 WANDB API Key: {api_key[:10]}..." if api_key else "❌ WANDB API Key not found")
    print(f"👤 WANDB Entity: {entity}")
    print(f"📊 Project Name: {project_name}")
    print("-" * 50)
    
    if not api_key:
        print("❌ WANDB_API_KEY is not set in environment variables")
        return False
    
    if not entity:
        print("❌ WANDB_ENTITY is not set in environment variables")
        return False
    
    try:
        # Initialize wandb
        print("🔄 Initializing WANDB...")
        wandb.login(key=api_key)
        
        # Test read permissions - try to access existing runs
        print("📖 Testing READ permissions...")
        api = wandb.Api()
        
        try:
            # Try to get project info
            project = api.project(name=project_name, entity=entity)
            print(f"✅ Successfully accessed project: {project.name}")
            
            # Try to list runs (read permission test)
            runs = list(api.runs(f"{entity}/{project_name}", per_page=5))
            print(f"✅ Successfully read runs. Found {len(runs)} runs (showing max 5)")
            
            for i, run in enumerate(runs[:3]):  # Show first 3 runs
                print(f"   📌 Run {i+1}: {run.name} ({run.state})")
                
        except Exception as e:
            print(f"⚠️  Could not access existing project/runs: {e}")
            print("   This might be normal if the project doesn't exist yet")
        
        # Test write permissions - create a test run
        print("\n✏️  Testing WRITE permissions...")
        
        # Initialize a test run
        test_run = wandb.init(
            project=project_name,
            entity=entity,
            name="test_permissions_run",
            tags=["test", "permissions"],
            notes="Test run to verify write permissions",
            mode="online"  # Ensure it actually syncs to wandb
        )
        
        # Log some test metrics
        test_run.log({"test_metric": 0.95, "epoch": 1})
        test_run.log({"test_loss": 0.05, "epoch": 2})
        
        print("✅ Successfully created test run and logged metrics")
        print(f"🔗 Run URL: {test_run.url}")
        
        # Finish the run
        test_run.finish()
        
        print("✅ Successfully finished test run")
        
        # Verify the run was created by listing recent runs
        print("\n🔍 Verifying run was created...")
        recent_runs = list(api.runs(f"{entity}/{project_name}", per_page=3))
        test_run_found = any(run.name == "test_permissions_run" for run in recent_runs)
        
        if test_run_found:
            print("✅ Test run found in recent runs - WRITE permission confirmed")
        else:
            print("⚠️  Test run not found in recent runs (might take time to sync)")
        
        return True
        
    except wandb.errors.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("   Check if your WANDB_API_KEY is correct")
        return False
        
    except wandb.errors.CommError as e:
        print(f"❌ Communication error: {e}")
        print("   Check your internet connection and API key")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def cleanup_test_runs():
    """Clean up test runs (optional)"""
    try:
        load_dotenv()
        api_key = os.getenv('WANDB_API_KEY')
        entity = os.getenv('WANDB_ENTITY')
        project_name = "vietnamese-legal-llama-unsloth"
        
        if not api_key or not entity:
            print("❌ Cannot cleanup - missing credentials")
            return
            
        wandb.login(key=api_key)
        api = wandb.Api()
        
        print("\n🧹 Looking for test runs to cleanup...")
        runs = api.runs(f"{entity}/{project_name}")
        
        test_runs = [run for run in runs if run.name == "test_permissions_run"]
        
        if test_runs:
            print(f"Found {len(test_runs)} test runs")
            choice = input("Do you want to delete test runs? (y/N): ").lower()
            
            if choice == 'y':
                for run in test_runs:
                    try:
                        run.delete()
                        print(f"✅ Deleted test run: {run.id}")
                    except Exception as e:
                        print(f"❌ Failed to delete run {run.id}: {e}")
            else:
                print("Skipping cleanup")
        else:
            print("No test runs found to cleanup")
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting WANDB permissions test...")
    print("=" * 60)
    
    success = test_wandb_permissions()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 WANDB permissions test completed successfully!")
        print("✅ You have both READ and WRITE access to the project")
        
        # Ask if user wants to cleanup
        print("\n" + "-" * 40)
        cleanup_choice = input("Do you want to cleanup test runs? (y/N): ").lower()
        if cleanup_choice == 'y':
            cleanup_test_runs()
    else:
        print("❌ WANDB permissions test failed!")
        print("Please check your API key and entity settings")
    
    print("\nTest completed.")