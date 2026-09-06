/* GhostLayer shared help center: static, client-side assistance for all public pages. */
(function () {
    'use strict';

    const helpArticles = [
        {
            title: 'What is GhostLayer in plain English?',
            tags: ['overview', 'basics', 'product'],
            body: 'GhostLayer is presented as a decision and memory layer for AI training infrastructure. The intended workflow is to observe workload behavior, record the context for a recommendation, apply only approved changes, and retain an audit trail for later review.'
        },
        {
            title: 'Are the performance figures production guarantees?',
            tags: ['results', 'benchmarks', 'proof', 'validation'],
            body: 'No. Treat performance figures as evidence with different confidence levels. The site labels some rows as physical-hardware measurements and others as synthetic projections. A production decision should rely on the exact workload, hardware, baseline, measurement window, policy thresholds, and raw evidence—not a headline metric.'
        },
        {
            title: 'What does “shadow mode” mean here?',
            tags: ['shadow mode', 'safety', 'deployment', 'rollout'],
            body: 'Shadow mode means observing and modelling a workload without making production changes. It is a sensible first stage for validating a baseline, comparing recommendations, defining rollback conditions, and obtaining change-control approval before any automated action is enabled.'
        },
        {
            title: 'How should I read the savings calculator?',
            tags: ['calculator', 'roi', 'cost', 'pricing'],
            body: 'The calculator is a planning estimate. It multiplies the GPUs, selected hourly rate, and monthly runtime, then applies the speedup you enter. The “net client savings” figure assumes a 20% performance-fee share. It is not a quote, invoice, or independently verified saving; replace all inputs with your contracted rates and measured baseline.'
        },
        {
            title: 'What is the difference between a demo and a live deployment?',
            tags: ['demo', 'digital twin', 'deployment', 'telemetry'],
            body: 'Interactive cluster panels and telemetry on this site are demonstrations unless a specific workload is identified as a physical-hardware measurement. A live deployment needs authenticated telemetry, a documented change-control policy, observable rollback tests, and a responsible operator.'
        },
        {
            title: 'What should a security review verify?',
            tags: ['security', 'privacy', 'vpc', 'compliance'],
            body: 'Do not rely on marketing labels alone. Ask for the deployment architecture, data-flow diagram, access controls, retention and deletion policy, encryption details, incident process, and current third-party attestations. Confirm whether every claimed control applies to the exact deployment model you plan to use.'
        },
        {
            title: 'How should we start an evaluation?',
            tags: ['evaluation', 'assessment', 'getting started', 'pilot'],
            body: 'Start with one representative training job and define success before tuning: baseline workload, required model-quality metric, allowed intervention scope, safety thresholds, rollback owner, evaluation duration, and who signs off. Expand only after the evidence is reproducible and the operational owners accept the risk controls.'
        },
        {
            title: 'Why can’t I send a request from this preview?',
            tags: ['contact', 'quote', 'form', 'support'],
            body: 'This is a static website preview. The request form is intentionally marked as non-transmitting until it is connected to a verified CRM, email service, or secure server endpoint. Before launch, configure an owned contact channel, privacy notice, consent text, spam protection, error handling, and a response workflow.'
        }
    ];

    let launcher;
    let panel;
    let search;
    let results;
    let status;
    let lastFocusedElement;

    function renderArticles(query) {
        const normalized = query.trim().toLowerCase();
        const matching = helpArticles.filter((article) => {
            const searchable = [article.title, article.body, article.tags.join(' ')].join(' ').toLowerCase();
            return !normalized || searchable.includes(normalized);
        });

        results.innerHTML = '';
        status.textContent = normalized
            ? `${matching.length} help topic${matching.length === 1 ? '' : 's'} found for “${query.trim()}”.`
            : `${helpArticles.length} practical help topics available.`;

        if (!matching.length) {
            results.innerHTML = '<div class="help-empty"><strong>No exact match.</strong><span>Try “calculator”, “security”, “pilot”, or “results”.</span></div>';
            return;
        }

        matching.forEach((article) => {
            const details = document.createElement('details');
            details.className = 'help-article';
            const summary = document.createElement('summary');
            summary.textContent = article.title;
            const body = document.createElement('p');
            body.textContent = article.body;
            details.append(summary, body);
            results.appendChild(details);
        });
    }

    function openHelp(topic) {
        lastFocusedElement = document.activeElement;
        panel.classList.add('is-open');
        launcher.setAttribute('aria-expanded', 'true');
        document.body.classList.add('help-open');
        if (topic) search.value = topic;
        renderArticles(search.value);
        window.setTimeout(() => search.focus(), 120);
    }

    function closeHelp() {
        panel.classList.remove('is-open');
        launcher.setAttribute('aria-expanded', 'false');
        document.body.classList.remove('help-open');
        if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') lastFocusedElement.focus();
    }

    function buildHelpCenter() {
        launcher = document.createElement('button');
        launcher.type = 'button';
        launcher.className = 'help-launcher font-mono';
        launcher.id = 'helpLauncher';
        launcher.setAttribute('aria-controls', 'helpPanel');
        launcher.setAttribute('aria-expanded', 'false');
        launcher.innerHTML = '<i class="fa-solid fa-circle-question" aria-hidden="true"></i><span>HELP</span>';

        panel = document.createElement('aside');
        panel.className = 'help-panel';
        panel.id = 'helpPanel';
        panel.setAttribute('aria-label', 'GhostLayer help center');
        panel.innerHTML = `
            <div class="help-panel-head">
                <div>
                    <p class="help-eyebrow font-mono"><i class="fa-solid fa-life-ring" aria-hidden="true"></i> PRODUCT HELP</p>
                    <h2>Get the context before you act.</h2>
                    <p class="help-intro">Plain-language guidance for the product, evidence, calculator, security review, and pilot planning.</p>
                </div>
                <button class="help-close" type="button" aria-label="Close help center"><i class="fa-solid fa-xmark" aria-hidden="true"></i></button>
            </div>
            <div class="help-search-wrap">
                <label class="sr-only" for="helpSearch">Search help topics</label>
                <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
                <input id="helpSearch" type="search" autocomplete="off" placeholder="Search help: calculator, security, pilot…">
            </div>
            <p id="helpStatus" class="help-status font-mono" aria-live="polite"></p>
            <div id="helpResults" class="help-results"></div>
            <div class="help-disclosure">
                <i class="fa-solid fa-circle-info" aria-hidden="true"></i>
                <p><strong>Important:</strong> This help center clarifies the website. It does not replace technical due diligence, a security review, or a written commercial agreement.</p>
            </div>
        `;

        document.body.append(panel, launcher);
        search = panel.querySelector('#helpSearch');
        results = panel.querySelector('#helpResults');
        status = panel.querySelector('#helpStatus');

        launcher.addEventListener('click', () => openHelp());
        panel.querySelector('.help-close').addEventListener('click', closeHelp);
        search.addEventListener('input', () => renderArticles(search.value));
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && panel.classList.contains('is-open')) closeHelp();
        });
        document.querySelectorAll('[data-help-open]').forEach((trigger) => {
            trigger.addEventListener('click', (event) => {
                event.preventDefault();
                openHelp(trigger.getAttribute('data-help-open') || '');
            });
        });
        renderArticles('');
    }

    document.addEventListener('DOMContentLoaded', buildHelpCenter);
}());
