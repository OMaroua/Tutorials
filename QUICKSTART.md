# Quick Start Guide - Tutorial Website

## Why You're Seeing Bare HTML

If you're seeing unstyled HTML pages, it's because you're opening the files directly in your browser (`file:///...`). This is a **Jekyll site** that needs to be served through Jekyll to work properly.

## How to Run the Site Locally

### Step 1: Install Jekyll (One-Time Setup)

First, make sure you have Ruby installed:

```bash
# Check if Ruby is installed
ruby -v

# If not installed, install Ruby:
# macOS (using Homebrew):
brew install ruby

# Linux (Ubuntu/Debian):
sudo apt-get install ruby-full

# Windows:
# Download from https://rubyinstaller.org/
```

### Step 2: Install Dependencies

Navigate to the Tutorials directory and install gems:

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials

# Install Bundler if you don't have it
gem install bundler

# Install all dependencies
bundle install
```

This will install Jekyll and all required plugins.

### Step 3: Serve the Website

```bash
# From the Tutorials directory, run:
bundle exec jekyll serve

# You should see output like:
# Server address: http://127.0.0.1:4000/Tutorials/
# Server running... press ctrl-c to stop.
```

### Step 4: View in Browser

Open your browser and go to:

```
http://localhost:4000/Tutorials/
```

🎉 **Now you should see the beautiful styled tutorial website!**

## Troubleshooting

### Issue: "bundle: command not found"

Solution:
```bash
gem install bundler
```

### Issue: Permission errors on macOS

Solution:
```bash
# Use the --user-install flag
gem install bundler --user-install

# Or use rbenv/rvm to manage Ruby versions
```

### Issue: Port 4000 already in use

Solution:
```bash
# Use a different port
bundle exec jekyll serve --port 4001
```

Then visit `http://localhost:4001/Tutorials/`

### Issue: CSS/styling still not loading

Make sure you're accessing:
- ✅ `http://localhost:4000/Tutorials/` (CORRECT)
- ❌ `file:///Users/.../index.html` (WRONG - no styling)

## Development Tips

### Auto-Reload

Jekyll watches for file changes automatically. Just edit your files and refresh the browser!

### Viewing on Mobile/Other Devices

If you want to view the site on your phone or tablet on the same network:

```bash
bundle exec jekyll serve --host=0.0.0.0
```

Then access using your computer's IP address:
```
http://192.168.1.XXX:4000/Tutorials/
```

## What Jekyll Does

Jekyll is a static site generator that:
1. Processes `.scss` files into `.css`
2. Applies layouts and templates
3. Converts Markdown to HTML
4. Handles includes and navigation
5. Serves everything with proper paths

That's why opening files directly doesn't work - Jekyll needs to process everything first!

## Alternative: Deploy to GitHub Pages

If you don't want to run Jekyll locally, just push to GitHub and enable GitHub Pages:

1. Push your code to GitHub
2. Go to Settings > Pages
3. Select the main branch as source
4. Your site will be live at `https://yourusername.github.io/Tutorials/`

GitHub Pages runs Jekyll automatically!

## Quick Commands Cheat Sheet

```bash
# Start the server
bundle exec jekyll serve

# Start with drafts visible
bundle exec jekyll serve --drafts

# Build without serving (output in _site/)
bundle exec jekyll build

# Clean generated files
bundle exec jekyll clean

# Stop the server
Ctrl+C
```

## Need Help?

- Jekyll Documentation: https://jekyllrb.com/docs/
- GitHub Pages Docs: https://docs.github.com/en/pages

---

**Happy Learning! 🚀**

