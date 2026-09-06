import zipfile
import os
import sys
from pathlib import Path

def build_all_archives():
    root = Path(__file__).resolve().parent.parent
    zips_dir = root / 'archive' / 'zips'
    zips_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory Pre-Export Gate: Verify Shipping Hygiene Before Generating Zips
    print("Executing mandatory pre-export shipping hygiene gate...", flush=True)
    import subprocess
    verify_script = root / "scripts" / "verify_shipping_hygiene.py"
    res = subprocess.run([sys.executable, str(verify_script)], cwd=str(root))
    if res.returncode != 0:
        raise RuntimeError("FATAL: Shipping hygiene verification failed! Aborting archive generation.")
    print("Shipping hygiene verified. Proceeding with archive bundling...\n", flush=True)

    exclude_dirs = {
        'node_modules', '.venv', '.gpu_env', '__pycache__', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', '.git', '.firebase', '.manus', 'dist', 'build',
        'zips', 'archive', 'scratch', '.agents'
    }

    # 1. Full Project Bundle (QuantGPUMgment_Project.zip)
    full_zip_path = zips_dir / 'QuantGPUMgment_Project.zip'
    print(f'Building {full_zip_path.name}...', flush=True)
    with zipfile.ZipFile(full_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for foldername, subfolders, filenames in os.walk(root):
            # Prune excluded directories in-place to avoid descending into them
            subfolders[:] = [d for d in subfolders if d not in exclude_dirs]
            
            for filename in filenames:
                if filename.endswith('.zip') or filename == 'QuantGPUMgment_Project.zip':
                    continue
                file_path = Path(foldername) / filename
                rel_file = os.path.relpath(file_path, root)
                zf.write(file_path, rel_file)

    # 2. GhostLayer Codebase + Docs + Tests (GhostLayer_Antigravity_Latest.zip)
    code_zip_path = zips_dir / 'GhostLayer_Antigravity_Latest.zip'
    print(f'Building {code_zip_path.name}...', flush=True)
    allowed_top_dirs = {'ghost_layer', 'tests', 'technical_docs', 'docs', 'scripts', 'examples', 'shared', 'client', 'server', 'charts', 'ceo_evidence_engine', '.ghostlayer'}
    allowed_root_files = {
        'README.md', 'pyproject.toml', 'package.json', 'IDEOLOGY.md', 'audit_report.md',
        'index.html', 'how-it-works.html', 'founder.html', 'app.js', 'style.css', 'help.js',
        'graph_report.html', 'variance_sphere_telemetry.png', 'TEST_INVENTORY.json',
        'tsconfig.json', 'tsconfig.node.json', 'vite.config.ts', 'components.json',
        'CLAUDE.md', 'GEMINI.md'
    }


    with zipfile.ZipFile(code_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for foldername, subfolders, filenames in os.walk(root):
            rel_folder = os.path.relpath(foldername, root)
            if rel_folder == '.':
                subfolders[:] = [d for d in subfolders if d in allowed_top_dirs and d not in exclude_dirs]
                for filename in filenames:
                    if filename in allowed_root_files:
                        file_path = root / filename
                        zf.write(file_path, filename)
            else:
                subfolders[:] = [d for d in subfolders if d not in exclude_dirs]
                for filename in filenames:
                    if filename.endswith('.zip'):
                        continue
                    file_path = Path(foldername) / filename
                    rel_file = os.path.relpath(file_path, root)
                    zf.write(file_path, rel_file)

    # 3. Technical Docs Bundle (ghostlayer_technical_docs.zip)
    docs_zip_path = zips_dir / 'ghostlayer_technical_docs.zip'
    print(f'Building {docs_zip_path.name}...', flush=True)
    with zipfile.ZipFile(docs_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for doc_dir in ['technical_docs', 'docs']:
            target_doc_dir = root / doc_dir
            if not target_doc_dir.exists():
                continue
            for foldername, subfolders, filenames in os.walk(target_doc_dir):
                subfolders[:] = [d for d in subfolders if d not in exclude_dirs]
                for filename in filenames:
                    file_path = Path(foldername) / filename
                    rel_file = os.path.relpath(file_path, root)
                    zf.write(file_path, rel_file)

    # 4. Website Showcase Bundle (GhostLayer_Website_Showcase.zip)
    website_zip_path = zips_dir / 'GhostLayer_Website_Showcase.zip'
    print(f'Building {website_zip_path.name}...', flush=True)
    website_dirs = {'client', 'server', 'shared', 'charts', 'assets', 'public'}
    website_files = {
        'index.html', 'founder.html', 'how-it-works.html', 'app.js', 'style.css',
        'help.js', 'graph_report.html', 'package.json', 'vite.config.ts',
        'tsconfig.json', 'tsconfig.node.json', 'components.json', 'README.md'
    }
    with zipfile.ZipFile(website_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in website_files:
            file_path = root / filename
            if file_path.exists():
                zf.write(file_path, filename)
        for dir_name in website_dirs:
            target_dir = root / dir_name
            if not target_dir.exists():
                continue
            for foldername, subfolders, filenames in os.walk(target_dir):
                subfolders[:] = [d for d in subfolders if d not in exclude_dirs]
                for filename in filenames:
                    file_path = Path(foldername) / filename
                    rel_file = os.path.relpath(file_path, root)
                    zf.write(file_path, rel_file)

    print('\nAll archives rebuilt successfully in archive/zips/:', flush=True)
    for p in [full_zip_path, code_zip_path, docs_zip_path, website_zip_path]:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f' - {p.name}: {size_mb:.2f} MB ({p.stat().st_size:,} bytes)', flush=True)

if __name__ == '__main__':
    build_all_archives()
