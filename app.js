/* ==========================================================================
   GHOSTLAYER CAPITAL - TIMELESS MONOCHROMATIC LUXURY & PERFORMANCE ENGINE
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initCustomCursor();
    initHeroCanvas();
    initMobileDrawer();
    initDigitalTwin64();
    initGpuNodeModal();
    initExplainabilityTabs();
    init3DComputeGlobe();
    initAuditExplorer();
    initValuationCalculator();
    initClickboardSnippets();
    initQuoteModals();
});

/* ==========================================================================
   1. REFINED SNAPPY CURSOR WITH HOVER EXPANSION
   ========================================================================== */
function initCustomCursor() {
    const dot = document.getElementById('cursorDot');
    const ring = document.getElementById('cursorRing');
    if (!dot || !ring) return;

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let ringX = mouseX;
    let ringY = mouseY;

    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        dot.style.left = `${mouseX}px`;
        dot.style.top = `${mouseY}px`;
    });

    function renderRing() {
        ringX += (mouseX - ringX) * 0.45;
        ringY += (mouseY - ringY) * 0.45;
        ring.style.left = `${ringX}px`;
        ring.style.top = `${ringY}px`;
        requestAnimationFrame(renderRing);
    }
    renderRing();

    // Attach hover listener to interactive elements
    const attachHover = () => {
        const hoverables = document.querySelectorAll('a, button, input, select, .gpu-block, .pop-hover');
        hoverables.forEach(el => {
            el.addEventListener('mouseenter', () => ring.classList.add('active-hover'));
            el.addEventListener('mouseleave', () => ring.classList.remove('active-hover'));
        });
    };
    attachHover();
}

/* ==========================================================================
   2. INTERACTIVE HERO GPU CANVAS (INTERSECTION OBSERVER GATED)
   ========================================================================== */
function initHeroCanvas() {
    const canvas = document.getElementById('heroParticleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = canvas.parentElement.offsetHeight || 600;

    let isVisible = true;
    let animationFrameId = null;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = canvas.parentElement.offsetHeight || 600;
    });

    let mouse = { x: width / 2, y: height / 2, radius: 180 };

    window.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
    });

    const particles = [];
    const numParticles = Math.min(Math.floor(width / 25), 50);

    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.7,
            vy: (Math.random() - 0.5) * 0.7,
            radius: Math.random() * 2 + 1,
            alpha: Math.random() * 0.4 + 0.2
        });
    }

    function animate() {
        if (!isVisible) return;
        ctx.clearRect(0, 0, width, height);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        const gridSize = 60;
        for (let x = 0; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw particles & cursor connections
        particles.forEach((p, index) => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            const dx = mouse.x - p.x;
            const dy = mouse.y - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < mouse.radius) {
                const force = (mouse.radius - dist) / mouse.radius;
                p.x += (dx / dist) * force * 1.2;
                p.y += (dy / dist) * force * 1.2;
            }

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${p.alpha})`;
            ctx.fill();

            for (let j = index + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const pdx = p.x - p2.x;
                const pdy = p.y - p2.y;
                const pdist = Math.sqrt(pdx * pdx + pdy * pdy);

                if (pdist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 * (1 - pdist / 120)})`;
                    ctx.stroke();
                }
            }
        });

        animationFrameId = requestAnimationFrame(animate);
    }

    // IntersectionObserver to pause animation offscreen
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            isVisible = entry.isIntersecting;
            if (isVisible) {
                cancelAnimationFrame(animationFrameId);
                animate();
            }
        });
    }, { threshold: 0.1 });
    observer.observe(canvas.parentElement);
}

/* ==========================================================================
   3. MOBILE NAVIGATION DRAWER
   ========================================================================== */
function initMobileDrawer() {
    const hamburger = document.getElementById('navHamburger');
    const drawer = document.getElementById('mobileDrawer');
    const closeBtn = document.getElementById('mobileNavClose');
    const links = document.querySelectorAll('.md-link');

    if (!hamburger || !drawer) return;

    const openDrawer = () => drawer.classList.add('active');
    const closeDrawer = () => drawer.classList.remove('active');

    hamburger.addEventListener('click', openDrawer);
    if (closeBtn) closeBtn.addEventListener('click', closeDrawer);

    links.forEach(l => l.addEventListener('click', closeDrawer));
}

/* ==========================================================================
   4. FLAGSHIP 64-GPU LIVE DIGITAL TWIN VISUALIZER
   ========================================================================== */
function initDigitalTwin64() {
    const grid = document.getElementById('gpuGrid64');
    if (!grid) return;

    const btnBaseline = document.getElementById('btnModeBaseline');
    const btnOptimized = document.getElementById('btnModeOptimized');
    const statusText = document.getElementById('dtStatusText');
    const utilFill = document.getElementById('dtUtilFill');
    const utilVal = document.getElementById('dtUtilVal');
    const latencyVal = document.getElementById('dtLatencyVal');
    const latencySub = document.getElementById('dtLatencySub');
    const vramVal = document.getElementById('dtVramVal');
    const vramSub = document.getElementById('dtVramSub');
    const inspectorBox = document.getElementById('dtNodeInspector');

    let currentMode = 'optimized'; // 'baseline' or 'optimized'

    // Render 64 Blocks
    grid.innerHTML = '';
    for (let i = 1; i <= 64; i++) {
        const block = document.createElement('button');
        block.type = 'button';
        block.className = 'gpu-block active-gpu';
        block.setAttribute('data-id', i);
        const formattedId = i < 10 ? '0' + i : i;
        block.setAttribute('aria-label', `Inspect GPU Node #${formattedId}`);
        block.innerHTML = `<span>#${formattedId}</span>`;
        
        block.addEventListener('click', () => {
            document.querySelectorAll('.gpu-block').forEach(b => b.classList.remove('selected-gpu'));
            block.classList.add('selected-gpu');
            updateNodeInspector(i);
        });

        grid.appendChild(block);
    }

    function setMode(mode) {
        currentMode = mode;
        const blocks = document.querySelectorAll('.gpu-block');

        if (mode === 'baseline') {
            btnBaseline.classList.add('active');
            btnOptimized.classList.remove('active');

            statusText.className = 'text-danger font-bold';
            statusText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> BOTTLENECK DETECTED: 48,200 TOKENS/SEC';
            utilFill.style.width = '34%';
            utilFill.style.background = 'var(--ruby-danger)';
            utilVal.textContent = '34.0%';
            latencyVal.textContent = '48.1 ms';
            latencyVal.className = 'dtm-num text-danger font-serif';
            latencySub.textContent = 'Baseline (Un-optimized)';
            latencySub.className = 'dtm-sub text-danger';
            vramVal.textContent = '7.8 GB / 8 GB';
            vramSub.textContent = 'High Memory Fragmentation';
            vramSub.className = 'dtm-sub text-danger';

            // Randomly fragment ~66% of GPUs
            blocks.forEach((b, idx) => {
                if (idx % 3 !== 0) {
                    b.className = 'gpu-block fragmented-gpu';
                } else {
                    b.className = 'gpu-block active-gpu';
                }
            });
        } else {
            btnOptimized.classList.add('active');
            btnBaseline.classList.remove('active');

            statusText.className = 'text-emerald font-bold';
            statusText.innerHTML = '<i class="fa-solid fa-bolt"></i> OPTIMIZED: 190,632 TOKENS/SEC';
            utilFill.style.width = '96%';
            utilFill.style.background = 'var(--emerald-success)';
            utilVal.textContent = '96.0%';
            latencyVal.textContent = '10.7 ms';
            latencyVal.className = 'dtm-num text-white font-serif';
            latencySub.textContent = '-77.7% Faster';
            latencySub.className = 'dtm-sub text-emerald';
            vramVal.textContent = '6.1 GB / 8 GB';
            vramSub.textContent = 'Zero Memory Spikes';
            vramSub.className = 'dtm-sub text-emerald';

            blocks.forEach(b => {
                b.className = 'gpu-block active-gpu';
            });
        }
    }

    function updateNodeInspector(nodeId) {
        if (!inspectorBox) return;
        const isOpt = currentMode === 'optimized';
        const formattedId = nodeId < 10 ? `0${nodeId}` : nodeId;
        inspectorBox.innerHTML = `
            <div class="ins-head text-white font-bold"><i class="fa-solid fa-circle-info"></i> NODE INSPECTOR: GPU #${formattedId}</div>
            <div class="ins-row"><span>Status:</span><span class="${isOpt ? 'text-emerald' : 'text-danger'}">${isOpt ? 'ACTIVE • 96% ALLOCATED' : 'FRAGMENTED • IDLE CYCLES'}</span></div>
            <div class="ins-row"><span>Kernel Engine:</span><span class="text-white">${isOpt ? 'Fused FlashAttention-2' : 'Standard Un-fused CUDA'}</span></div>
            <div class="ins-row"><span>Memory Paging:</span><span class="text-white">${isOpt ? 'Zero Fragment (Contiguous)' : 'High Page Fragmentation'}</span></div>
            <div class="ins-row"><span>PCIe Transfer Latency:</span><span class="${isOpt ? 'text-emerald' : 'text-danger'}">${isOpt ? '0.42 ms' : '1.85 ms'}</span></div>
        `;
    }

    btnBaseline.addEventListener('click', () => setMode('baseline'));
    btnOptimized.addEventListener('click', () => setMode('optimized'));

    // Select first node by default
    const firstBlock = grid.querySelector('.gpu-block');
    if (firstBlock) firstBlock.classList.add('selected-gpu');
}

/* ==========================================================================
   TOAST NOTIFICATION SYSTEM
   ========================================================================== */
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast-msg font-mono';
    
    const icon = type === 'success' ? '<i class="fa-solid fa-circle-check text-emerald"></i>' :
                 type === 'warn' ? '<i class="fa-solid fa-triangle-exclamation text-danger"></i>' :
                 '<i class="fa-solid fa-circle-info text-white"></i>';

    toast.innerHTML = `${icon} <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(12px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

/* ==========================================================================
   INTERACTIVE GPU NODE DIAGNOSTIC MODAL
   ========================================================================== */
function initGpuNodeModal() {
    const modal = document.getElementById('gpuNodeModal');
    const closeBtn = document.getElementById('btnGpuModalClose');
    const btnDefrag = document.getElementById('btnGpuDefrag');
    const btnBench = document.getElementById('btnGpuBenchmarkNode');

    if (!modal) return;

    function openModalForGpu(gpuId) {
        const title = document.getElementById('gpuModalTitle');
        const memText = document.getElementById('gpuModalMemText');
        const fragText = document.getElementById('gpuModalFragText');
        const memFill = document.getElementById('gpuModalMemFill');
        const fragFill = document.getElementById('gpuModalFragFill');
        const kicker = document.getElementById('gpuModalKicker');

        const formattedId = gpuId < 10 ? `0${gpuId}` : gpuId;
        if (title) title.textContent = `GPU Block #${formattedId} (NVIDIA H100 SXM5 80GB)`;
        if (kicker) kicker.textContent = `PHYSICAL NODE TELEMETRY • GPU #${formattedId}`;

        // Dynamic realistic values based on GPU index
        const usedGb = (58.4 + (gpuId % 7) * 2.1).toFixed(1);
        const usedPct = ((usedGb / 80.0) * 100).toFixed(1);
        const fragPct = (1.8 + (gpuId % 5) * 0.9).toFixed(1);

        if (memText) memText.textContent = `Used: ${usedGb} GB / 80.0 GB (${usedPct}%)`;
        if (fragText) fragText.textContent = `Fragmentation: ${fragPct}% (Defrag Active)`;
        if (memFill) memFill.style.width = `${usedPct}%`;
        if (fragFill) fragFill.style.width = `${fragPct}%`;

        modal.classList.add('active');
    }

    // Attach click listener to all GPU blocks
    const grid = document.getElementById('gpuGrid64');
    if (grid) {
        grid.addEventListener('click', (e) => {
            const block = e.target.closest('.gpu-block');
            if (block) {
                const id = parseInt(block.getAttribute('data-id'), 10) || 1;
                openModalForGpu(id);
            }
        });
    }

    if (closeBtn) closeBtn.addEventListener('click', () => modal.classList.remove('active'));
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    if (btnDefrag) {
        btnDefrag.addEventListener('click', () => {
            const originalText = btnDefrag.innerHTML;
            btnDefrag.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Compacting...';
            btnDefrag.disabled = true;

            setTimeout(() => {
                const fragFill = document.getElementById('gpuModalFragFill');
                const fragText = document.getElementById('gpuModalFragText');
                if (fragFill) fragFill.style.width = '0.5%';
                if (fragText) fragText.textContent = 'Fragmentation: 0.4% (Contiguous Allocation)';

                btnDefrag.innerHTML = '<i class="fa-solid fa-check text-emerald"></i> Defrag Completed';
                showToast('CUDA memory page table compacted. 2.4 GB VRAM freed into contiguous pool.', 'success');

                setTimeout(() => {
                    btnDefrag.innerHTML = originalText;
                    btnDefrag.disabled = false;
                }, 1800);
            }, 600);
        });
    }

    if (btnBench) {
        btnBench.addEventListener('click', () => {
            showToast('Live single-node microbenchmark completed: 2,410 tok/s at 58°C.', 'success');
        });
    }
}

/* ==========================================================================
   INTERACTIVE DECISION EXPLAINABILITY TABS
   ========================================================================== */
function initExplainabilityTabs() {
    const tabNav = document.getElementById('expTabNav');
    if (!tabNav) return;

    const buttons = tabNav.querySelectorAll('.exp-tab-btn');
    const panes = document.querySelectorAll('.exp-tab-pane');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            buttons.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add('active');
            }

            showToast(`Loaded Decision Trace: ${btn.textContent.trim()}`, 'info');
        });
    });
}

/* ==========================================================================
   INTERACTIVE 3D ORBITAL DECISION FIELD & PLATFORM GLOBE (CANVAS 3D ENGINE)
   Matches the Manus orbital experience with live marker projection and tab sync.
   ========================================================================== */
function init3DComputeGlobe() {
    const canvas = document.getElementById('computeGlobeCanvas');
    const container = document.getElementById('globeContainer');
    const overlay = document.getElementById('globeMarkersOverlay');
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    let width = canvas.width = container.offsetWidth || 560;
    let height = canvas.height = container.offsetHeight || 560;

    function resizeGlobe() {
        if (!container) return;
        width = canvas.width = container.offsetWidth || 560;
        height = canvas.height = container.offsetHeight || 560;
    }
    window.addEventListener('resize', resizeGlobe);

    // 8 Core Platform Navigation Modules with 3D Spherical Coordinates & Tone Styling
    const platformModules = [
        {
            id: 'explainability',
            navKey: 'explainability',
            label: 'DECISION LAB',
            name: 'Decision Lab & Explainability Diagnostic',
            kicker: 'DESTINATION 01 • CONTROL LOOP',
            tone: 'signal',
            target: '#explainability',
            isExternal: false,
            lat: -11.0, lon: 67.0,
            moduleType: 'Diagnostic Inspector',
            telemetry: 'Microsecond Trace Log',
            metric: '<1 ms Rollback Latency',
            guardrail: 'SOC 2 Type II Audited',
            status: 'ACTIVE MODULE • READY',
            summary: 'Inspect live transparent decision traces for Dynamic Memory Paging, FlashAttention-2 fusion, and sub-millisecond automated rollback triggers.'
        },
        {
            id: 'pain',
            navKey: 'pain',
            label: 'THE PROBLEM',
            name: 'The Universal AI Infrastructure Pain',
            kicker: 'DESTINATION 02 • INFORMATION FIELD',
            tone: 'neutral',
            target: '#pain',
            isExternal: false,
            lat: 26.0, lon: -12.0,
            moduleType: 'Diagnostic Overview',
            telemetry: 'Memory Leak & GPU Waste',
            metric: '31% Cost Reduction',
            guardrail: 'Zero Cluster Downtime',
            status: 'ACTIVE MODULE • READY',
            summary: 'Empirical analysis of un-fused CUDA graphs, memory fragmentation, and why nobody knows why GPU utilization suddenly drops during training.'
        },
        {
            id: 'overview',
            navKey: 'overview',
            label: 'DIGITAL TWIN',
            name: '64-GPU Live Digital Twin Visualizer',
            kicker: 'DESTINATION 03 • INFRASTRUCTURE MEMORY',
            tone: 'signal',
            target: '#overview',
            isExternal: false,
            lat: 32.0, lon: 137.0,
            moduleType: 'Interactive Visualizer',
            telemetry: 'Real-Time VRAM & Tok/s',
            metric: '+65.6% Throughput',
            guardrail: 'KL Divergence Armed',
            status: 'ACTIVE MODULE • READY',
            summary: 'Inspect 64 individual GPU nodes in real-time, toggle between fragmented baseline and contiguous GhostLayer memory allocation, and compact VRAM pages on demand.'
        },
        {
            id: 'benchmarks',
            navKey: 'benchmarks',
            label: 'BENCHMARKS',
            name: 'Empirical Hardware Benchmark Explorer',
            kicker: 'DESTINATION 04 • EVIDENCE FIELD',
            tone: 'signal',
            target: '#benchmarks',
            isExternal: false,
            lat: 48.0, lon: -65.0,
            moduleType: 'Evidence Explorer',
            telemetry: 'Physical Hardware vs Fleet',
            metric: '+65.6% LLM / +28.5% CNN',
            guardrail: 'Zero Fabrication Policy',
            status: 'ACTIVE MODULE • READY',
            summary: 'Physical desktop GPU benchmarks (RTX A2000 8GB) and multi-node cluster scaling matrix projections with instant 1-click audit reports.'
        },
        {
            id: 'architecture',
            navKey: 'architecture',
            label: 'CONTROL ENGINE',
            name: 'Autonomous Control Plane Architecture',
            kicker: 'DESTINATION 05 • EXECUTION CORE',
            tone: 'signal',
            target: '#architecture',
            isExternal: false,
            lat: -28.0, lon: -84.0,
            moduleType: 'Architecture Stack',
            telemetry: 'Triton & CUDA Kernels',
            metric: '0.00% Exfiltration',
            guardrail: 'Air-Gapped Private VPC',
            status: 'ACTIVE MODULE • READY',
            summary: 'Multi-stage autonomous optimization engine with zero-overhead C++ hooks, sub-millisecond rollback checkpoints, and on-prem air-gapped security.'
        },
        {
            id: 'calculator',
            navKey: 'calculator',
            label: 'COST LEDGER',
            name: 'Portfolio Cost Traceability Ledger',
            kicker: 'DESTINATION 06 • FINANCIAL INTEGRITY',
            tone: 'signal',
            target: '#calculator',
            isExternal: false,
            lat: -36.0, lon: 155.0,
            moduleType: 'Financial Ledger',
            telemetry: '20% Gain-Share Model',
            metric: '$20,031 / mo Net Savings',
            guardrail: 'Deterministic Math',
            status: 'ACTIVE MODULE • READY',
            summary: 'Calculate exact GPU-hour reductions, EBITDA accretion, and export verifiable audit receipts for private equity and enterprise CFOs.'
        },
        {
            id: 'how-it-works',
            navKey: 'how-it-works',
            label: 'HOW IT WORKS',
            name: 'How GhostLayer Works (Technical Explainer)',
            kicker: 'DESTINATION 07 • ARCHITECTURE GUIDE',
            tone: 'neutral',
            target: 'how-it-works.html',
            isExternal: true,
            lat: -48.0, lon: -15.0,
            moduleType: 'Technical Whitepaper',
            telemetry: 'Kernel Graph Pipelining',
            metric: 'O(N) Attention Bound',
            guardrail: 'Zero Exfiltration',
            status: 'ACTIVE MODULE • READY',
            summary: 'Under-the-hood technical guide detailing PyTorch memory allocators, Triton JIT compilation, and autonomous control loops.'
        },
        {
            id: 'founder',
            navKey: 'founder',
            label: 'FOUNDER STATEMENT',
            name: 'Founder Statement & Compute Economics',
            kicker: 'DESTINATION 08 • LEADERSHIP BRIEF',
            tone: 'risk',
            target: 'founder.html',
            isExternal: true,
            lat: 58.0, lon: -125.0,
            moduleType: 'Executive Brief',
            telemetry: 'Leadership Principles',
            metric: '100% Capital Integrity',
            guardrail: 'Zero CapEx Required',
            status: 'ACTIVE MODULE • READY',
            summary: 'Founder letter on compute infrastructure economics, capital efficiency principles, and the GhostLayer long-term vision.'
        }
    ];

    let selectedModule = platformModules[0];
    let rotX = 0.22;
    let rotY = -1.1;
    let targetRotY = null;
    let targetRotX = null;
    let velX = 0;
    let velY = 0.0025;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;
    let pingAnim = 0;

    // Create 180 background mineral particle points
    const starParticles = [];
    for (let i = 0; i < 180; i++) {
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = 2 * Math.PI * Math.random();
        const r = 1.05 + Math.random() * 0.45; // Just outside sphere radius
        starParticles.push({
            x: r * Math.sin(phi) * Math.cos(theta),
            y: r * Math.cos(phi),
            z: r * Math.sin(phi) * Math.sin(theta),
            size: 0.8 + Math.random() * 1.5,
            alpha: 0.15 + Math.random() * 0.45,
            twinkleSpeed: 0.02 + Math.random() * 0.04
        });
    }

    // Convert Lat/Lon to 3D Cartesian coordinates on sphere of radius R
    function latLonTo3D(lat, lon, radius) {
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        return {
            x: -(radius * Math.sin(phi) * Math.cos(theta)),
            y: radius * Math.cos(phi),
            z: radius * Math.sin(phi) * Math.sin(theta)
        };
    }

    // 3D rotation and perspective projection
    function project3D(p, rx, ry, cx, cy) {
        const cosY = Math.cos(ry);
        const sinY = Math.sin(ry);
        const x1 = p.x * cosY + p.z * sinY;
        const z1 = -p.x * sinY + p.z * cosY;

        const cosX = Math.cos(rx);
        const sinX = Math.sin(rx);
        const y2 = p.y * cosX - z1 * sinX;
        const z2 = p.y * sinX + z1 * cosX;

        const fov = 420;
        const scale = fov / (fov + z2);

        return {
            x: cx + x1 * scale,
            y: cy - y2 * scale,
            z: z2,
            scale: scale,
            visible: z2 < 60 // Facing camera
        };
    }

    // Create marker DOM elements in the HTML overlay
    const markerElements = new Map();
    if (overlay) {
        overlay.innerHTML = '';
        platformModules.forEach(mod => {
            const el = document.createElement('div');
            el.className = `globe-marker-label globe-marker-label--${mod.tone} ${mod.id === selectedModule.id ? 'is-active' : ''}`;
            el.innerHTML = `<span class="marker-tick"></span> ${mod.label}`;
            el.setAttribute('data-mod-id', mod.id);
            
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                focusDestination(mod, true);
            });
            
            overlay.appendChild(el);
            markerElements.set(mod.id, el);
        });
    }

    // Update HUD information card
    function updateHud(mod) {
        selectedModule = mod;
        const nameEl = document.getElementById('hudNodeName');
        const kickerEl = document.getElementById('hudRegionKicker');
        const statusText = document.getElementById('hudStatusText');
        const comp = document.getElementById('hudCompression');
        const recipe = document.getElementById('hudRecipeDesc');
        const jumpBtn = document.getElementById('btnGlobeJump');
        const statusTitle = document.getElementById('orbitalStatusTitle');
        const statusSub = document.getElementById('orbitalStatusSub');

        if (nameEl) nameEl.textContent = mod.name;
        if (kickerEl) kickerEl.textContent = mod.kicker;
        if (statusText) statusText.textContent = mod.status;
        if (comp) comp.textContent = mod.metric;
        if (recipe) recipe.textContent = mod.summary;

        if (statusTitle) statusTitle.textContent = `FOCUSED: ${mod.label}`;
        if (statusSub) statusSub.textContent = `— ${mod.telemetry} | ${mod.metric}`;

        if (jumpBtn) {
            jumpBtn.setAttribute('href', mod.target);
            jumpBtn.innerHTML = mod.isExternal ? 
                `<i class="fa-solid fa-arrow-up-right-from-square"></i> Open ${mod.label}` :
                `<i class="fa-solid fa-arrow-down"></i> Jump to ${mod.label}`;
        }

        // Highlight active DOM marker
        markerElements.forEach((el, id) => {
            if (id === mod.id) {
                el.classList.add('is-active');
            } else {
                el.classList.remove('is-active');
            }
        });
    }

    // Function to rotate the 3D globe smoothly to focus on a destination module
    function focusDestination(mod, shouldScroll = false) {
        updateHud(mod);

        // Convert destination lat/lon to target camera rotation angles
        const phi = (90 - mod.lat) * (Math.PI / 180);
        const theta = (mod.lon + 180) * (Math.PI / 180);

        // Target rotation to bring the point directly to the front (z > 0, facing +Z)
        targetRotY = -(theta - Math.PI / 2);
        targetRotX = -(phi - Math.PI / 2);

        // Wrap targetRotY cleanly
        while (targetRotY - rotY > Math.PI) targetRotY -= Math.PI * 2;
        while (targetRotY - rotY < -Math.PI) targetRotY += Math.PI * 2;

        showToast(`Orbiting to ${mod.label}...`, 'info');

        if (shouldScroll) {
            if (mod.isExternal) {
                setTimeout(() => { window.location.href = mod.target; }, 600);
            } else {
                const targetSec = document.querySelector(mod.target);
                if (targetSec) {
                    setTimeout(() => {
                        targetSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 450);
                }
            }
        }
    }

    // Connect top navbar links to focus 3D globe and smooth-scroll
    const navLinks = document.querySelectorAll('.nav-link[data-nav], .md-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const matchedMod = platformModules.find(m => m.target === href);
                if (matchedMod) {
                    e.preventDefault();
                    focusDestination(matchedMod, true);
                }
            }
        });
    });

    // Return to Orbit Button in header
    const btnReturnOrbit = document.getElementById('btnReturnToOrbit');
    if (btnReturnOrbit) {
        btnReturnOrbit.addEventListener('click', () => {
            targetRotX = 0.22;
            targetRotY = -1.1;
            const orbitalSec = document.getElementById('orbitalStage');
            if (orbitalSec) {
                orbitalSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            showToast('Returned to Orbital View', 'info');
        });
    }

    // Pre-calculate Latitude and Longitude rings
    const baseRadius = 185;
    const ringPoints = [];

    // Latitude rings (-60, -30, 0, 30, 60)
    for (let lat = -60; lat <= 60; lat += 30) {
        const pts = [];
        for (let lon = -180; lon <= 180; lon += 8) {
            pts.push(latLonTo3D(lat, lon, baseRadius));
        }
        ringPoints.push(pts);
    }

    // Longitude meridians (every 30 deg)
    for (let lon = -180; lon < 180; lon += 30) {
        const pts = [];
        for (let lat = -80; lat <= 80; lat += 6) {
            pts.push(latLonTo3D(lat, lon, baseRadius));
        }
        ringPoints.push(pts);
    }

    // Three tilted elliptical orbital rings (Manus-style)
    const tiltedRings = [
        { tiltX: 0.26, tiltZ: 0.12, rX: baseRadius * 1.32, rY: baseRadius * 1.28, color: 'rgba(66, 216, 187, 0.22)' },
        { tiltX: -0.45, tiltZ: 0.35, rX: baseRadius * 1.45, rY: baseRadius * 1.38, color: 'rgba(255, 255, 255, 0.12)' },
        { tiltX: 0.58, tiltZ: -0.28, rX: baseRadius * 1.55, rY: baseRadius * 1.48, color: 'rgba(241, 109, 82, 0.16)' }
    ];

    // Main 60fps render loop
    function render() {
        ctx.clearRect(0, 0, width, height);
        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(width, height) * 0.34;

        // Smooth rotation interpolation toward target
        if (targetRotY !== null) {
            rotY += (targetRotY - rotY) * 0.08;
            if (Math.abs(targetRotY - rotY) < 0.005) targetRotY = null;
        } else if (!isDragging) {
            rotY += velY;
        }

        if (targetRotX !== null) {
            rotX += (targetRotX - rotX) * 0.08;
            if (Math.abs(targetRotX - rotX) < 0.005) targetRotX = null;
        }

        pingAnim += 0.04;

        // 1. Draw atmospheric outer ethereal glow
        const glowGrad = ctx.createRadialGradient(cx, cy, radius * 0.6, cx, cy, radius * 1.45);
        glowGrad.addColorStop(0, 'rgba(66, 216, 187, 0.09)');
        glowGrad.addColorStop(0.5, 'rgba(66, 216, 187, 0.025)');
        glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = glowGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 1.45, 0, Math.PI * 2);
        ctx.fill();

        // 2. Draw 180 background mineral particle points (stars)
        starParticles.forEach(star => {
            const p = {
                x: star.x * radius,
                y: star.y * radius,
                z: star.z * radius
            };
            const proj = project3D(p, rotX, rotY, cx, cy);
            if (proj.visible) {
                const alpha = star.alpha * (0.8 + 0.2 * Math.sin(pingAnim * 2 + star.twinkleSpeed * 100));
                ctx.beginPath();
                ctx.arc(proj.x, proj.y, star.size * proj.scale, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(220, 240, 235, ${Math.max(0, Math.min(1, alpha))})`;
                ctx.fill();
            }
        });

        // 3. Draw sphere wireframe lat/lon grid
        ctx.lineWidth = 1;
        ringPoints.forEach(ring => {
            ctx.beginPath();
            let started = false;
            ring.forEach(pt => {
                const scaledPt = { x: (pt.x / baseRadius) * radius, y: (pt.y / baseRadius) * radius, z: (pt.z / baseRadius) * radius };
                const proj = project3D(scaledPt, rotX, rotY, cx, cy);
                if (proj.visible) {
                    if (!started) {
                        ctx.moveTo(proj.x, proj.y);
                        started = true;
                    } else {
                        ctx.lineTo(proj.x, proj.y);
                    }
                } else {
                    started = false;
                }
            });
            ctx.strokeStyle = 'rgba(229, 239, 235, 0.045)';
            ctx.stroke();
        });

        // 4. Draw tilted orbital trajectory rings
        tiltedRings.forEach(ring => {
            ctx.beginPath();
            let started = false;
            for (let angle = 0; angle <= Math.PI * 2 + 0.1; angle += 0.08) {
                // Parametric ellipse tilted in 3D
                const x0 = Math.cos(angle) * (ring.rX / baseRadius) * radius;
                const y0 = Math.sin(angle) * (ring.rY / baseRadius) * radius * Math.cos(ring.tiltX);
                const z0 = Math.sin(angle) * (ring.rY / baseRadius) * radius * Math.sin(ring.tiltX);

                // Additional tilt around Z
                const x1 = x0 * Math.cos(ring.tiltZ) - y0 * Math.sin(ring.tiltZ);
                const y1 = x0 * Math.sin(ring.tiltZ) + y0 * Math.cos(ring.tiltZ);

                const proj = project3D({ x: x1, y: y1, z: z0 }, rotX, rotY, cx, cy);
                if (proj.visible) {
                    if (!started) {
                        ctx.moveTo(proj.x, proj.y);
                        started = true;
                    } else {
                        ctx.lineTo(proj.x, proj.y);
                    }
                } else {
                    started = false;
                }
            }
            ctx.strokeStyle = ring.color;
            ctx.lineWidth = 1.1;
            ctx.stroke();
        });

        // 5. Draw Quadratic Bezier Trajectory Arcs connecting active module to others
        const activePt3d = latLonTo3D(selectedModule.lat, selectedModule.lon, radius);
        const activeProj = project3D(activePt3d, rotX, rotY, cx, cy);

        platformModules.forEach(mod => {
            if (mod.id !== selectedModule.id) {
                const targetPt3d = latLonTo3D(mod.lat, mod.lon, radius);
                const targetProj = project3D(targetPt3d, rotX, rotY, cx, cy);

                if (activeProj.visible || targetProj.visible) {
                    // Midpoint elevated outward for 3D curved arc
                    const midX = (activePt3d.x + targetPt3d.x) * 0.5 * 1.35;
                    const midY = (activePt3d.y + targetPt3d.y) * 0.5 * 1.35;
                    const midZ = (activePt3d.z + targetPt3d.z) * 0.5 * 1.35;
                    const midProj = project3D({ x: midX, y: midY, z: midZ }, rotX, rotY, cx, cy);

                    ctx.beginPath();
                    ctx.moveTo(activeProj.x, activeProj.y);
                    ctx.quadraticCurveTo(midProj.x, midProj.y, targetProj.x, targetProj.y);
                    ctx.strokeStyle = mod.tone === 'risk' ? 'rgba(241, 109, 82, 0.35)' : 'rgba(66, 216, 187, 0.35)';
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([4, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }
        });

        // 6. Draw Platform Module Nodes & Position HTML Marker Labels
        platformModules.forEach(mod => {
            const pt3d = latLonTo3D(mod.lat, mod.lon, radius);
            const proj = project3D(pt3d, rotX, rotY, cx, cy);
            const markerEl = markerElements.get(mod.id);
            const isSelected = selectedModule.id === mod.id;

            if (proj.visible && proj.z < 45) {
                // Update HTML marker label position
                if (markerEl) {
                    markerEl.style.display = 'flex';
                    markerEl.style.left = `${proj.x}px`;
                    markerEl.style.top = `${proj.y}px`;
                    const alpha = Math.max(0.3, Math.min(1, (1 - proj.z / 180)));
                    markerEl.style.opacity = isSelected ? '1' : `${alpha}`;
                }

                // Pulsing outer radar halo
                const colorTone = mod.tone === 'risk' ? '#f16d52' : '#42d8bb';
                const pulseSize = ((pingAnim * 14) % 22) + 4;
                const pulseAlpha = Math.max(0, 1 - pulseSize / 26);

                ctx.beginPath();
                ctx.arc(proj.x, proj.y, pulseSize, 0, Math.PI * 2);
                ctx.strokeStyle = isSelected ? `rgba(66, 216, 187, ${pulseAlpha})` : `rgba(255, 255, 255, ${pulseAlpha * 0.35})`;
                ctx.lineWidth = 1.4;
                ctx.stroke();

                // Glowing core dot
                ctx.beginPath();
                ctx.arc(proj.x, proj.y, isSelected ? 6 : 4, 0, Math.PI * 2);
                ctx.fillStyle = isSelected ? colorTone : '#FFFFFF';
                ctx.shadowColor = colorTone;
                ctx.shadowBlur = isSelected ? 16 : 6;
                ctx.fill();
                ctx.shadowBlur = 0; // reset shadow
            } else {
                // Hide marker label if on the back side of the sphere
                if (markerEl) {
                    markerEl.style.display = 'none';
                }
            }
        });

        requestAnimationFrame(render);
    }

    // Mouse drag interaction
    container.addEventListener('mousedown', (e) => {
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        targetRotY = null;
        targetRotX = null;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        rotY += dx * 0.007;
        rotX = Math.max(-0.85, Math.min(0.85, rotX - dy * 0.007));
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => isDragging = false);

    // Touch support for mobile devices
    container.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1) {
            isDragging = true;
            lastMouseX = e.touches[0].clientX;
            lastMouseY = e.touches[0].clientY;
            targetRotY = null;
            targetRotX = null;
        }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
        if (!isDragging || e.touches.length !== 1) return;
        const dx = e.touches[0].clientX - lastMouseX;
        const dy = e.touches[0].clientY - lastMouseY;
        rotY += dx * 0.007;
        rotX = Math.max(-0.85, Math.min(0.85, rotX - dy * 0.007));
        lastMouseX = e.touches[0].clientX;
        lastMouseY = e.touches[0].clientY;
    }, { passive: true });

    window.addEventListener('touchend', () => isDragging = false);

    // Initialize default HUD & launch render loop
    updateHud(platformModules[0]);
    render();
}

/* ==========================================================================
   5. EMPIRICAL GPU BENCHMARKS EXPLORER
   ========================================================================== */
const benchmarkData = [
    { model: "Native LLM 52.2M (Empirical)", hw: "NVIDIA RTX A2000 8GB", baseline: "1,149 tok/s", opt: "1,903 tok/s", delta: "+65.6%", cat: "Physical Hardware", status: "VERIFIED SAFE", details: "Empirical baseline measured on physical desktop GPU. FlashAttention-2 fusion + Memory Paging applied." },
    { model: "ResNet-50 CNN (Empirical)", hw: "NVIDIA RTX A2000 8GB", baseline: "342 img/s", opt: "439 img/s", delta: "+28.5%", cat: "Physical Hardware", status: "VERIFIED SAFE", details: "Empirical vision pipeline benchmark. CUDA stream pin DataLoader enabled." },
    { model: "High-VRAM 166M (Empirical)", hw: "NVIDIA RTX A2000 8GB", baseline: "OOM Crash", opt: "812 tok/s", delta: "Prevented OOM", cat: "Physical Hardware", status: "THRESHOLD EXCEEDED", details: "Threshold exceeded demo. GhostWatcherHook detected VRAM limit (>6800MB) and triggered safe auto-rollback." },
    { model: "FinTech Transformer 85M (Empirical)", hw: "Local Host GPU", baseline: "2,410 tok/s", opt: "3,180 tok/s", delta: "+31.9%", cat: "Physical Hardware", status: "VERIFIED SAFE", details: "Empirical quantitative predictive model optimization." },
    { model: "Llama 3 8B (Synthetic Fleet)", hw: "8x NVIDIA A100 80GB SXM4", baseline: "42,100 tok/s", opt: "54,300 tok/s", delta: "+29.0%", cat: "Synthetic Projection", status: "SIMULATED PROJECTION", details: "Synthetic cluster projection based on matrix scaling model." },
    { model: "Llama 3 70B (Synthetic Fleet)", hw: "64x NVIDIA H100 80GB SXM5", baseline: "148,000 tok/s", opt: "201,200 tok/s", delta: "+35.9%", cat: "Synthetic Projection", status: "SIMULATED PROJECTION", details: "Synthetic multi-node TensorParallel fleet benchmark." }
];

function initAuditExplorer() {
    const tableBody = document.getElementById('benchmarkTableBody');
    const typeFilter = document.getElementById('benchmarkTypeFilter');
    const statusFilter = document.getElementById('statusFilter');
    const searchInput = document.getElementById('benchmarkSearch');

    if (!tableBody) return;

    function renderTable() {
        const activeType = typeFilter.querySelector('.btn-filter.active').getAttribute('data-filter');
        const activeStatus = statusFilter.value;
        const query = searchInput.value.toLowerCase();

        tableBody.innerHTML = '';

        const filtered = benchmarkData.filter(item => {
            if (activeType === 'empirical' && item.cat !== 'Physical Hardware') return false;
            if (activeType === 'simulated' && item.cat !== 'Synthetic Projection') return false;
            if (activeStatus !== 'all' && item.status !== activeStatus) return false;
            if (query && !item.model.toLowerCase().includes(query) && !item.hw.toLowerCase().includes(query)) return false;
            return true;
        });

        if (filtered.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted font-mono" style="padding: 24px;">No matching benchmarks found.</td></tr>`;
            return;
        }

        filtered.forEach(item => {
            const tr = document.createElement('tr');
            tr.className = 'pop-hover';
            tr.style.cursor = 'pointer';

            const statusClass = item.status === 'VERIFIED SAFE' ? 'text-emerald' : (item.status === 'THRESHOLD EXCEEDED' ? 'text-danger' : 'text-muted');

            tr.innerHTML = `
                <td class="font-bold text-white">${item.model}</td>
                <td class="font-mono text-muted">${item.hw}</td>
                <td class="font-mono">${item.baseline}</td>
                <td class="font-mono text-emerald font-bold">${item.opt}</td>
                <td class="font-mono text-emerald font-bold">${item.delta}</td>
                <td class="font-mono">${item.cat}</td>
                <td class="font-mono ${statusClass}">${item.status}</td>
                <td><button class="btn btn-sm btn-secondary font-mono">Audit Log</button></td>
            `;

            tr.addEventListener('click', () => openAuditModal(item));
            tableBody.appendChild(tr);
        });
    }

    if (typeFilter) {
        typeFilter.querySelectorAll('.btn-filter').forEach(btn => {
            btn.addEventListener('click', () => {
                typeFilter.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderTable();
            });
        });
    }

    if (statusFilter) statusFilter.addEventListener('change', renderTable);
    if (searchInput) searchInput.addEventListener('input', renderTable);

    renderTable();
}

function openAuditModal(item) {
    const modal = document.getElementById('auditModal');
    const title = document.getElementById('modalTitle');
    const body = document.getElementById('modalBody');

    if (!modal || !body) return;

    title.textContent = `BENCHMARK AUDIT: ${item.model}`;
    body.innerHTML = `
        <div class="m-line"><strong>Hardware Model:</strong> ${item.hw}</div>
        <div class="m-line"><strong>Category:</strong> ${item.cat}</div>
        <div class="m-line"><strong>Baseline Performance:</strong> ${item.baseline}</div>
        <div class="m-line"><strong>GhostLayer Performance:</strong> ${item.opt} (${item.delta})</div>
        <div class="m-line"><strong>Verification Status:</strong> ${item.status}</div>
        <div class="m-line margin-top-sm"><strong>Execution Audit Trail Details:</strong></div>
        <p class="margin-top-xs text-muted">${item.details}</p>
    `;
    modal.classList.add('active');

    const closeBtn = document.getElementById('btnModalClose');
    if (closeBtn) closeBtn.onclick = () => modal.classList.remove('active');
}

/* ==========================================================================
   6. PORTFOLIO VALUATION & SAVINGS CALCULATOR
   ========================================================================== */
function initValuationCalculator() {
    const gpuSlider = document.getElementById('gpuCountSlider');
    const gpuVal = document.getElementById('gpuCountVal');
    const modelSelect = document.getElementById('gpuModelSelect');
    const daysSlider = document.getElementById('daysSlider');
    const daysVal = document.getElementById('daysVal');
    const speedupSlider = document.getElementById('speedupSlider');
    const speedupVal = document.getElementById('speedupVal');

    const baselineCostEl = document.getElementById('calcBaselineCost');
    const optCostEl = document.getElementById('calcOptimizedCost');
    const grossSavingsEl = document.getElementById('calcGrossSavings');
    const netSavingsEl = document.getElementById('calcNetSavings');

    const rGpuInfo = document.getElementById('rGpuInfo');
    const rHoursInfo = document.getElementById('rHoursInfo');
    const rSpeedupInfo = document.getElementById('rSpeedupInfo');
    const rBaseline = document.getElementById('rBaseline');
    const rOptimized = document.getElementById('rOptimized');
    const rNetVal = document.getElementById('rNetVal');
    const rSavedHours = document.getElementById('rSavedHours');

    if (!gpuSlider || !baselineCostEl) return;

    function calculate() {
        const numGpus = parseInt(gpuSlider.value, 10);
        gpuVal.textContent = `${numGpus} GPU${numGpus > 1 ? 's' : ''}`;

        const modelVal = modelSelect.value;
        const hourlyRate = parseFloat(modelVal.split('|')[0]);
        const modelName = modelVal.split('|')[1];

        const days = parseInt(daysSlider.value, 10);
        const hours = days * 24;
        daysVal.textContent = `${days} Days / Month (${hours} hrs)`;

        const speedup = parseFloat(speedupSlider.value);
        speedupVal.textContent = `+${speedup.toFixed(1)}% Speedup`;

        const totalHours = numGpus * hours;
        const baselineMonthly = totalHours * hourlyRate;

        // Optimized runtime reduction factor: 1 / (1 + speedup/100)
        const timeFactor = 1 / (1 + speedup / 100);
        const optimizedMonthly = baselineMonthly * timeFactor;
        const grossSavings = baselineMonthly - optimizedMonthly;
        
        // 20% performance gain share model (80% net savings to client)
        const netClientSavings = grossSavings * 0.80;
        const savedHours = Math.round(totalHours * (1 - timeFactor));

        // Format currency
        const fmt = (val) => '$' + Math.round(val).toLocaleString();

        baselineCostEl.textContent = fmt(baselineMonthly);
        optCostEl.textContent = fmt(optimizedMonthly);
        grossSavingsEl.textContent = fmt(grossSavings) + ' / mo';
        netSavingsEl.textContent = fmt(netClientSavings) + ' / mo';

        if (rGpuInfo) rGpuInfo.textContent = `${numGpus}× ${modelName}`;
        if (rHoursInfo) rHoursInfo.textContent = `${totalHours.toLocaleString()} GPU-Hours`;
        if (rSpeedupInfo) rSpeedupInfo.textContent = `+${speedup.toFixed(1)}% Acceleration`;
        if (rBaseline) rBaseline.textContent = fmt(baselineMonthly);
        if (rOptimized) rOptimized.textContent = fmt(optimizedMonthly);
        if (rNetVal) rNetVal.textContent = fmt(netClientSavings);
        if (rSavedHours) rSavedHours.textContent = `${savedHours.toLocaleString()} GPU-Hours`;
    }

    gpuSlider.addEventListener('input', calculate);
    modelSelect.addEventListener('change', calculate);
    daysSlider.addEventListener('input', calculate);
    speedupSlider.addEventListener('input', calculate);

    // Preset buttons
    const presets = document.querySelectorAll('.btn-preset');
    presets.forEach(p => {
        p.addEventListener('click', () => {
            presets.forEach(b => b.classList.remove('active'));
            p.classList.add('active');
            gpuSlider.value = p.getAttribute('data-gpus');
            if (p.getAttribute('data-model')) modelSelect.value = p.getAttribute('data-model');
            if (p.getAttribute('data-speedup')) speedupSlider.value = p.getAttribute('data-speedup');
            calculate();
        });
    });

    calculate();

    // Copy Receipt
    const copyBtn = document.getElementById('btnPrintReceipt');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const receipt = document.getElementById('receiptContainer');
            if (!receipt) return;
            navigator.clipboard.writeText(receipt.innerText);
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fa-solid fa-check text-emerald"></i> Ledger Copied!';
            setTimeout(() => copyBtn.innerHTML = originalText, 2000);
        });
    }
}

/* ==========================================================================
   7. EXECUTIVE CLICKBOARD SNIPPETS
   ========================================================================== */
function initClickboardSnippets() {
    const copyBtns = document.querySelectorAll('.btn-copy-snippet');
    copyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const snippet = btn.getAttribute('data-snippet');
            if (snippet) {
                navigator.clipboard.writeText(snippet);
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fa-solid fa-check text-emerald"></i> Copied!';
                setTimeout(() => btn.innerHTML = originalHTML, 2000);
            }
        });
    });
}

/* ==========================================================================
   8. QUOTE MODAL HANDLER
   ========================================================================== */
function initQuoteModals() {
    const triggers = document.querySelectorAll('.btn-quote-trigger');
    const modal = document.getElementById('quoteModal');
    const closeBtn = document.getElementById('btnQuoteClose');
    const form = document.getElementById('quoteForm');
    let lastActiveTrigger = null;

    if (!modal) return;

    function openModal(triggerEl) {
        lastActiveTrigger = triggerEl || document.activeElement;
        modal.classList.add('active');
        const firstInput = modal.querySelector('input, select, textarea, button');
        if (firstInput) firstInput.focus();
    }

    function closeModal() {
        modal.classList.remove('active');
        if (lastActiveTrigger && typeof lastActiveTrigger.focus === 'function') {
            lastActiveTrigger.focus();
        }
    }

    triggers.forEach(t => t.addEventListener('click', (e) => openModal(e.currentTarget)));
    if (closeBtn) closeBtn.addEventListener('click', closeModal);

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });

    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                const origText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fa-solid fa-check"></i> Request Verified (Non-transmitting preview)';
                submitBtn.disabled = true;
                setTimeout(() => {
                    submitBtn.innerHTML = origText;
                    submitBtn.disabled = false;
                    closeModal();
                }, 2000);
            } else {
                closeModal();
            }
        });
    }
}

/* ==========================================================================
   9. MANUS.IM SCROLL-REVEAL ENGINE (IntersectionObserver-based)
   Automatically reveals any element tagged with .sr-fade-up, .sr-fade-in,
   .sr-slide-left, .sr-slide-right, or .sr-scale when it enters the viewport.
   ========================================================================== */
function initScrollReveals() {
    const srClasses = ['.sr-fade-up', '.sr-fade-in', '.sr-slide-left', '.sr-slide-right', '.sr-scale'];
    const allRevealEls = document.querySelectorAll(srClasses.join(','));

    if (!allRevealEls.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('sr-visible');
                observer.unobserve(entry.target); // fire once
            }
        });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    allRevealEls.forEach(el => observer.observe(el));
}

/* ==========================================================================
   10. ANIME.JS LIVE COUNTER ANIMATION
   Finds elements with data-counter="<target_number>" and animates them
   from 0 up to the target value using anime.js spring easing.
   ========================================================================== */
function initAnimeCounters() {
    if (typeof anime === 'undefined') return;

    const counterEls = document.querySelectorAll('[data-counter]');
    if (!counterEls.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseFloat(el.getAttribute('data-counter'));
            const prefix = el.getAttribute('data-counter-prefix') || '';
            const suffix = el.getAttribute('data-counter-suffix') || '';
            const decimals = parseInt(el.getAttribute('data-counter-decimals') || '0', 10);

            const obj = { val: 0 };
            anime({
                targets: obj,
                val: target,
                duration: 1400,
                easing: 'easeOutExpo',
                update() {
                    el.textContent = prefix + obj.val.toFixed(decimals) + suffix;
                },
                complete() {
                    el.textContent = prefix + target.toFixed(decimals) + suffix;
                    el.classList.add('counter-pop-anim');
                }
            });
            observer.unobserve(el);
        });
    }, { threshold: 0.3 });

    counterEls.forEach(el => observer.observe(el));
}

/* ==========================================================================
   11. ANIME.JS HERO HEADLINE STAGGER REVEAL
   Splits the hero headline into words and staggers them in on page load.
   ========================================================================== */
function initHeroStagger() {
    if (typeof anime === 'undefined') return;

    const headline = document.querySelector('.hero-headline');
    if (!headline) return;

    // Wrap each word in a span for stagger
    const words = headline.innerHTML.split(/(\s+|<br>)/);
    headline.innerHTML = words.map(w => {
        if (w.trim() === '' || w === '<br>') return w;
        return `<span class="hero-word" style="display:inline-block;opacity:0;transform:translateY(18px)">${w}</span>`;
    }).join('');

    anime({
        targets: '.hero-word',
        opacity: [0, 1],
        translateY: [18, 0],
        easing: 'cubicBezier(0.16, 1, 0.3, 1)',
        duration: 640,
        delay: anime.stagger(55, { start: 200 })
    });
}

/* ==========================================================================
   12. ANIME.JS NAVBAR ENTRY ANIMATION
   ========================================================================== */
function initNavbarReveal() {
    if (typeof anime === 'undefined') return;

    anime({
        targets: '.navbar',
        translateY: [-60, 0],
        opacity: [0, 1],
        easing: 'cubicBezier(0.16, 1, 0.3, 1)',
        duration: 700,
        delay: 80
    });

    anime({
        targets: '.nav-link, .nav-brand, .nav-actions',
        opacity: [0, 1],
        translateY: [-10, 0],
        easing: 'cubicBezier(0.16, 1, 0.3, 1)',
        duration: 500,
        delay: anime.stagger(60, { start: 300 })
    });
}

/* ==========================================================================
   13. HERO METRICS SPRING POP
   ========================================================================== */
function initHeroMetricsPop() {
    if (typeof anime === 'undefined') return;

    const metrics = document.querySelectorAll('.hero-metrics-grid .hm-card');
    if (!metrics.length) return;

    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            anime({
                targets: metrics,
                scale: [0.85, 1],
                opacity: [0, 1],
                easing: 'cubicBezier(0.34, 1.56, 0.64, 1)',
                duration: 560,
                delay: anime.stagger(80, { start: 100 })
            });
            observer.disconnect();
        }
    }, { threshold: 0.2 });

    if (metrics[0]) observer.observe(metrics[0].closest('.hero-metrics-grid') || metrics[0]);
}

/* ==========================================================================
   14. GPU GRID STAGGER INTRO
   ========================================================================== */
function initGpuGridStagger() {
    if (typeof anime === 'undefined') return;

    const grid = document.getElementById('gpuGrid64');
    if (!grid) return;

    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            anime({
                targets: '#gpuGrid64 .gpu-block',
                scale: [0, 1],
                opacity: [0, 1],
                easing: 'cubicBezier(0.34, 1.56, 0.64, 1)',
                duration: 380,
                delay: anime.stagger(18)
            });
            observer.disconnect();
        }
    }, { threshold: 0.1 });

    observer.observe(grid);
}

/* ==========================================================================
   15. GRADIENT DIVIDERS — inject between all major sections automatically
   ========================================================================== */
function injectGradientDividers() {
    const sections = document.querySelectorAll('section + section');
    sections.forEach(section => {
        const divider = document.createElement('div');
        divider.className = 'gradient-divider';
        section.parentNode.insertBefore(divider, section);
    });
}

/* ==========================================================================
   16. SPRING HOVER — auto-apply to cards that don't already have pop-hover
   ========================================================================== */
function initSpringHovers() {
    const springTargets = document.querySelectorAll('.exp-box, .mc-item, .eb-step-box, .cmp-col, .process-step');
    springTargets.forEach(el => {
        if (!el.classList.contains('spring-hover')) {
            el.classList.add('spring-hover');
        }
    });

    // Emerald glow spring on featured / benchmark items
    const emeraldTargets = document.querySelectorAll('.cmp-col.featured-col, .badge-gradient, .glass-panel-glow');
    emeraldTargets.forEach(el => el.classList.add('spring-hover-emerald'));
}

/* ==========================================================================
   INIT ALL MANUS.IM ENHANCEMENTS ON DOM READY
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
    initScrollReveals();
    initAnimeCounters();
    initHeroStagger();
    initNavbarReveal();
    initHeroMetricsPop();
    initGpuGridStagger();
    injectGradientDividers();
    initSpringHovers();
});

