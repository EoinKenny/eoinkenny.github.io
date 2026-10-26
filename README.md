# Eoin Kenny's website

Static site, published by GitHub Pages from `main`. No installation or build is needed to serve the pages.

## Add a blog post

1. Create `blog/your-post-slug/index.html` and put its images in that folder's `assets/` directory. Keep the Home / Blog post navigation and give the page its own canonical URL.
2. Add an entry to `blog/posts.json`, following the existing entry.
3. Rebuild the listing and RSS feed with Python 3.9 or later:

   ```sh
   python3 scripts/build_blog.py
   ```

4. Commit the new post and generated files, then push to `main`. GitHub Pages publishes them automatically.

Posts appear newest first. Edit a published post directly in its `index.html`; the listing builder does not overwrite it.
