import time
import random
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase

def run_simulation():
    print("\n[INIT] Booting Global Ghost Layer Data Mind...")
    kb = SharedKnowledgeBase("ghost_simulation_db.json")
    
    print("\n--- PHASE 1: THE FREE TRIALS (Building the Data Mind) ---")
    clients = [
        {"name": "YC Startup Alpha", "arch": "Llama-3-8B", "gpus": 8, "hw": "8x H100"},
        {"name": "YC Startup Beta", "arch": "Llama-3-8B", "gpus": 16, "hw": "16x H100"},
        {"name": "Mid-Market Med-AI", "arch": "Mistral-7B", "gpus": 32, "hw": "32x A100"},
        {"name": "Open-Source Collective", "arch": "Llama-3-70B", "gpus": 64, "hw": "64x H100"},
        {"name": "stealth_agi_startup", "arch": "Mixture-of-Experts", "gpus": 128, "hw": "128x H100"}
    ]
    
    for i, client in enumerate(clients):
        print(f"\n[{client['name']}] Starting 'Insignificant' Free Trial...")
        time.sleep(0.5)
        
        improvement_pct = random.uniform(15.0, 30.0)
        opt_config = {"rule_id": "kernel_fusion", "params": {"block_size": 256, "dynamic_chunking": True}}
        
        kb.register_learning(
            architecture_family=client['arch'],
            hardware_type=client['hw'],
            effective_config=opt_config,
            throughput_improvement_pct=improvement_pct
        )
        print(f"  -> Ghost Layer mapped execution graph. Identified {improvement_pct:.1f}% inefficiency.")
        print(f"  -> [DATA MIND] Telemetry extracted. New optimization vector learned for {client['arch']} on {client['hw']}.")
    
    print("\n--- PHASE 2: THE WHALE CLIENT (Exploiting the Data Mind) ---")
    print("\n[Meta / xAI Competitor] Starting Massive Production Run...")
    whale_arch = "Llama-3-70B"
    whale_hw = "64x H100" # Matching the Open-Source Collective
    print(f"  -> Whale Architecture: {whale_arch} on {whale_hw}")
    
    print("  -> Ghost Layer querying Data Mind for pre-trained optimizations...")
    time.sleep(1)
    
    # Query the knowledge base
    historical_match = kb.query(architecture_family=whale_arch, hardware_type=whale_hw)
    
    if historical_match:
        print("  -> [DATA MIND HIT] Found exact match from 'Open-Source Collective' free trial!")
        print(f"  -> Applying Zero-Shot Optimization Config: {historical_match.effective_config}")
        print(f"  -> Immediate Expected Gain: {historical_match.throughput_improvement_pct:.1f}% reduction in compute time.")
        print("  -> Result: The Whale saves $1.2M on day one with zero risk. We take our 25% cut.")
    else:
        print("  -> No match found. Starting from scratch.")

    print("\n[SYSTEM] Simulation Complete. The Data Mind flywheel is active.")

if __name__ == "__main__":
    run_simulation()
