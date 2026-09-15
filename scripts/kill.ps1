import subprocess
import sys

def kill_ibgateway():
    """Kill IB Gateway/IBC process on this machine"""
    try:
        # Try killing javauser or IBGateway processes
        subprocess.run(["taskkill", "/F", "/IM", "java.exe"], 
                      capture_output=True, timeout=5)
        print("✓ Killed java.exe (IB Gateway)")
    except Exception as e:
        print(f"Failed to kill java.exe: {e}")
    
    try:
        subprocess.run(["taskkill", "/F", "/IM", "IBGateway.exe"], 
                      capture_output=True, timeout=5)
        print("✓ Killed IBGateway.exe")
    except Exception as e:
        print(f"Failed to kill IBGateway.exe: {e}")

if __name__ == "__main__":
    kill_ibgateway()
    print("Done")