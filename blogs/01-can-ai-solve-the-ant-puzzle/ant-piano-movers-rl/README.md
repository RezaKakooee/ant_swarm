# Can an AI agent solve the piano-movers puzzle that ants solve?

Project page for the single-agent piano-movers experiment: an RL agent learns to
thread a T-shaped load through two narrow slits — the maneuver crazy ants perform
collectively in Dreyer et al., PNAS 2025.

Static page, no build step. Drop this folder into the site repo (for example as
`ant-piano-movers-rl/`) and it is live at `rezakakooee.github.io/ant-piano-movers-rl/`.

    index.html     the post
    styles.css     styling (light, Geist + Geist Mono)
    assets/        figures, clips (mp4 + gif) and the ant video
    .nojekyll      serve files as-is

Source of the text: `blogs/01_single_agent/POST.md` in
[RezaKakooee/ant_swarm](https://github.com/RezaKakooee/ant-piano-movers-rl).
Figures are regenerated with `blogs/01_single_agent/make_figures.py`.
