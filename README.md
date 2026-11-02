## wrafell.github.io

Personal site of Willmer Quiñones — AI researcher, postdoctoral fellow at UNIST LAIT, working on representation learning, multimodal learning, and computational histopathology.

Plain Jekyll, no theme: cyanotype-inspired pixel art on ivory paper.

- `_layouts/` — `default` (shell), `about` (home page), `post`, `page`
- `_pages/about.md` — home page content (`/`); `_pages/writing.html` — post index (`/writing/`)
- `_posts/` — writing; set `mathjax: true` in front matter for LaTeX
- `assets/css/main.css` — all styles; palette tokens at the top
- `_tools/make_pixel_art.py` — regenerates the pixel images in `assets/images/` (needs Pillow + NumPy)

Local preview: `bundle install && bundle exec jekyll serve`
