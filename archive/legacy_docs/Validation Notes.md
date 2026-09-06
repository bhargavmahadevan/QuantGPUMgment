# Validation Notes

## Initial browser inspection

The initial landing-page inspection showed an overcrowded desktop navigation bar, a marketing-first hero, no visible support affordance, and a content-heavy, jargon-dense interface.

## Post-edit browser inspection

The second browser navigation appeared to serve stale cached HTML: its extracted labels still showed the pre-edit benchmark counts and prior receipt date even though the local source files contain the revised content. The next check must use a cache-busting URL and verify the added Help control, revised labels, and page-level script loading directly.

## Cache-busted home-page validation

A cache-busted preview confirmed the revised benchmark labels: **Physical Hardware (4) [LIMITED VALIDATION]** and **Synthetic Projections (2) [EXPERIMENTAL]**. The page focus confirmed the help launcher was present in the current build. A browser-side interaction invoked the launcher; the next visual check will verify the panel state and search behavior.

## Help-center visual validation

The help content and all eight articles rendered, but the browser screenshot exposed a layout defect: the panel appeared in document flow beneath the footer instead of as a fixed right-side overlay. This requires a CSS correction before the feature can be accepted.

## Cache-safe overlay correction

After versioning the stylesheet and scripts, the updated home preview rendered the Help launcher as a fixed bottom-right control. The digital-twin nodes also appeared as correctly labeled buttons rather than non-semantic clickable divs, confirming the updated interaction script loaded.

## Help overlay validation

The Help control now opens as a fixed, right-side overlay with eight concise expandable topics, a search field, a visible disclosure, and a close control. The underlying page remains visible while the panel is open. The next checks will verify filtering and the revised form response.

## Help search validation

Searching for **calculator** reduced the help center to one relevant topic, confirming client-side filtering works. The Help drawer was then closed and the revised evaluation form was opened for final form-flow validation.

## Evaluation-form validation

The evaluation form displayed explicit labels, a preview non-transmission notice, and the renamed **Validate Evaluation Request** action. It was dismissed successfully with the Escape key. No data was submitted during validation.

## Dialog keyboard fix

The dialog behavior was corrected in the shared script: opening now records the trigger and moves focus into the form; closing returns focus to the trigger. A cache-busted build loaded the correction and the evaluation dialog was opened for the final Escape-key check.

## Final dialog validation

The revised evaluation dialog now closes with the Escape key and returns focus to the request trigger. The earlier Escape-key failure was corrected before packaging.
