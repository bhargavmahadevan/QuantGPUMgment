import asyncio
import json
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="[Fleet Runner] %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Assume the KB path is local
LOCAL_KB_PATH = Path("ghost_knowledge_base.json")

async def execute_on_node(host: str, user: str, pem_file: str, remote_dir: str, model_id: str):
    """
    Executes the GhostLayer heavy benchmark on a remote GPU node via SSH.
    """
    logger.info(f"Deploying benchmark to {host} ({user})...")
    
    # Construct SSH command to run the script remotely
    ssh_cmd = [
        "ssh", "-i", pem_file, "-o", "StrictHostKeyChecking=no", f"{user}@{host}",
        f"cd {remote_dir} && ./.venv/bin/python examples/run_heavy_llama_benchmark.py --model_id {model_id} --steps 100 --batch_size 4"
    ]
    
    process = await asyncio.create_subprocess_exec(
        *ssh_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        logger.info(f"Node {host} completed successfully!")
        
        # Pull the report back
        scp_cmd = [
            "scp", "-i", pem_file, "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_dir}/real_training_validation_heavy_llama_report.md",
            f"fleet_reports/report_{host}.md"
        ]
        
        scp_process = await asyncio.create_subprocess_exec(*scp_cmd)
        await scp_process.communicate()
        
        # Pull the kb diff back
        scp_kb_cmd = [
            "scp", "-i", pem_file, "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_dir}/ghost_knowledge_base.json",
            f"fleet_reports/kb_{host}.json"
        ]
        scp_kb_process = await asyncio.create_subprocess_exec(*scp_kb_cmd)
        await scp_kb_process.communicate()
        
        return True
    else:
        logger.error(f"Node {host} failed with exit code {process.returncode}")
        logger.error(f"Stderr: {stderr.decode()}")
        return False

def merge_knowledge_bases():
    """
    Merges all retrieved JSON reports back into the central ghost_knowledge_base.json.
    """
    logger.info("Merging telemetry from all nodes into master Knowledge Base...")
    if not LOCAL_KB_PATH.exists():
        master_data = {}
    else:
        with open(LOCAL_KB_PATH, "r") as f:
            master_data = json.load(f)
            
    reports_dir = Path("fleet_reports")
    merged_count = 0
    
    for kb_file in reports_dir.glob("kb_*.json"):
        with open(kb_file, "r") as f:
            remote_data = json.load(f)
            
        for hw_key, remote_stats in remote_data.items():
            if hw_key not in master_data:
                master_data[hw_key] = remote_stats
                merged_count += remote_stats.get("sample_count", 0)
            else:
                # Basic Welford aggregation or simple replacement (for prototype)
                # In a real merge, we'd use Welford's parallel algorithm. 
                # Here we simply append the new runs for demonstration.
                master_stats = master_data[hw_key]
                new_n = master_stats["sample_count"] + remote_stats["sample_count"]
                master_stats["throughput_improvement_pct"] = (
                    (master_stats["throughput_improvement_pct"] * master_stats["sample_count"]) +
                    (remote_stats["throughput_improvement_pct"] * remote_stats["sample_count"])
                ) / new_n
                master_stats["sample_count"] = new_n
                merged_count += remote_stats["sample_count"]
                
    with open(LOCAL_KB_PATH, "w") as f:
        json.dump(master_data, f, indent=4)
        
    logger.info(f"Merged {merged_count} new empirical runs into the master KB.")

async def main():
    Path("fleet_reports").mkdir(exist_ok=True)
    
    # Define your cluster of 50 remote GPU nodes here
    # (e.g. AWS EC2 ips, RunPod IDs, etc)
    cluster_nodes = [
        # {"host": "198.51.100.14", "user": "ubuntu", "pem": "a100-key.pem", "dir": "~/QuantGPUMgment"},
        # {"host": "198.51.100.15", "user": "ubuntu", "pem": "a100-key.pem", "dir": "~/QuantGPUMgment"},
    ]
    
    if not cluster_nodes:
        logger.warning("No nodes configured in cluster_nodes array. Add your SSH targets to run the fleet.")
        return
        
    logger.info(f"Triggering GhostLayer benchmarks across {len(cluster_nodes)} nodes...")
    
    tasks = []
    for node in cluster_nodes:
        tasks.append(execute_on_node(
            host=node["host"],
            user=node["user"],
            pem_file=node["pem"],
            remote_dir=node["dir"],
            model_id="meta-llama/Llama-3.2-1B"
        ))
        
    results = await asyncio.gather(*tasks)
    successes = sum(1 for r in results if r)
    logger.info(f"Fleet execution complete. {successes}/{len(cluster_nodes)} succeeded.")
    
    if successes > 0:
        merge_knowledge_bases()

if __name__ == "__main__":
    asyncio.run(main())
