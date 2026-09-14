# Can a swarm of AI ants solve the puzzle?

Project page for part two of the ant piano-movers experiment: five and ten RL
agents, each seeing only its own small view, learn to carry a T-shaped load
through two narrow slits with no leader and no messages.

Static page, no build step. Drop this folder into the site repo (for example as
`ant-swarm-rl/`) and it is live at `rezakakooee.github.io/ant-swarm-rl/`.

    index.html     the post
    styles.css     styling (same as part one)
    assets/        figures and clips (mp4 + gif)
    .nojekyll      serve files as-is

Source of the text: `blogs/02-can-a-swarm-solve-the-ant-puzzle/POST.md` in
[RezaKakooee/ant_swarm](https://github.com/RezaKakooee/ant-piano-movers-rl).
Figures: `make_figures.py`. Clips: `render_clips.py` (fresh seeds 80000+).
