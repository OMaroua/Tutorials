# GitHub Pages Setup Guide

This guide will help you publish your tutorials as a beautiful website using GitHub Pages.

## Step 1: Create the Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it: `Tutorials`
5. Make it **Public** (required for free GitHub Pages)
6. **Do NOT** initialize with README, .gitignore, or license (we already have them)
7. Click "Create repository"

## Step 2: Push Your Local Code to GitHub

Open Terminal and run these commands from the `Tutorials` directory:

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: VAE tutorial with GitHub Pages setup"

# Add remote repository (replace OMaroua with your GitHub username)
git remote add origin https://github.com/OMaroua/Tutorials.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Enable GitHub Pages

1. Go to your repository on GitHub: `https://github.com/OMaroua/Tutorials`
2. Click on **Settings** (gear icon)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **Save**

GitHub will start building your site. This takes 1-2 minutes.

## Step 4: Access Your Website

Your site will be available at:

```
https://OMaroua.github.io/Tutorials/
```

The VAE tutorial specifically will be at:

```
https://OMaroua.github.io/Tutorials/01-VAE-Tutorial/tutorial.html
```

## Step 5: Verify Everything Works

1. Visit your site URL
2. Check that:
   - The homepage loads with the tutorial list
   - Navigation works
   - The VAE tutorial loads
   - Math equations render properly (LaTeX via MathJax)
   - Code blocks have syntax highlighting
   - Table of contents appears on the right (desktop view)

## Updating Your Website

Whenever you make changes to your tutorials:

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials

# Make your edits...

# Stage changes
git add .

# Commit
git commit -m "Update: describe your changes here"

# Push to GitHub
git push origin main
```

GitHub Pages will automatically rebuild your site (takes 1-2 minutes).

## Troubleshooting

### Issue: 404 Page Not Found

**Solution:** 
- Wait 2-3 minutes after enabling Pages
- Check that GitHub Pages is enabled in Settings → Pages
- Verify the branch is set to `main` and folder to `/ (root)`

### Issue: Math equations not rendering

**Solution:**
- Equations use `$...$` for inline and `$$...$$` for display
- MathJax takes a moment to load, refresh the page
- Check browser console for JavaScript errors

### Issue: Styling looks broken

**Solution:**
- GitHub Pages uses Jekyll to build the site
- The `_config.yml` file controls the configuration
- Check that `assets/css/style.scss` exists
- Clear browser cache and reload

### Issue: Changes not appearing

**Solution:**
- Check the Actions tab on GitHub to see if the build succeeded
- Wait 1-2 minutes after pushing
- Clear browser cache (Cmd+Shift+R on Mac)

## Customization

### Changing the Theme Color

Edit `assets/css/style.scss`:

```scss
:root {
    --primary-color: #4a1a8a;     /* Change this */
    --accent-color: #7c3aed;      /* And this */
}
```

### Updating Site Information

Edit `_config.yml`:

```yaml
title: "Your Title"
description: "Your description"
author: "Your Name"
email: "your.email@example.com"
```

### Adding More Tutorials

1. Create a new folder: `02-YourTutorial/`
2. Add a `tutorial.md` file with front matter:
```markdown
---
layout: tutorial
title: "Your Tutorial Title"
author: Your Name
date: 2025-11-09
---

# Your Tutorial Content
```
3. Update `index.md` to link to the new tutorial
4. Commit and push to GitHub

## Benefits of GitHub Pages

- **Free hosting** for your educational content
- **Automatic deployment** from your repository
- **Version control** via Git
- **Professional appearance** with custom styling
- **LaTeX support** for mathematical equations
- **Syntax highlighting** for code blocks
- **Responsive design** works on mobile and desktop
- **Easy to update** just push to GitHub

## Local Testing (Optional)

To preview your site locally before pushing:

```bash
# Install Jekyll (one-time setup)
gem install bundler jekyll

# Create Gemfile in Tutorials directory
cat > Gemfile << EOF
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
gem "webrick"
EOF

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Visit http://localhost:4000/Tutorials/
```

---

Your tutorials are now a professional website! Share the link with students, colleagues, and on your resume.

