# GitHub Pages Not Building - Diagnostic Guide

## Current Issue

The site is showing raw HTML instead of the styled Jekyll site.

## Quick Fixes to Try

### Fix 1: Check GitHub Pages is Enabled

1. Go to https://github.com/OMaroua/Tutorials
2. Click **Settings** → **Pages**  
3. Make sure:
   - Source is set to **Deploy from a branch**
   - Branch is set to **main** (not master)
   - Folder is set to **/ (root)**
4. Click **Save** if you made any changes
5. Wait 2-3 minutes and check the site

### Fix 2: Check for Build Errors

1. Go to https://github.com/OMaroua/Tutorials/actions
2. Look for any failed builds (red X icons)
3. Click on the most recent build to see error messages
4. Share the error message if you see one

### Fix 3: Force Rebuild

1. Make a small change to any file (add a space somewhere)
2. Commit and push
3. This will trigger a new build

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials
echo " " >> README.md
git add README.md
git commit -m "Trigger rebuild"
git push origin main
```

### Fix 4: Verify the Site URL

Make sure you're visiting the correct URL:
- ✅ CORRECT: `https://omaroua.github.io/Tutorials/`
- ❌ WRONG: `https://omaroua.github.io/Tutorials/index.md`
- ❌ WRONG: `https://github.com/OMaroua/Tutorials`

## Common Issues

### Issue: "404 - Page Not Found"
**Solution**: GitHub Pages is not enabled. Follow Fix 1 above.

### Issue: Raw HTML/Markdown showing
**Causes:**
1. GitHub Pages is not enabled
2. Jekyll build is failing
3. Wrong URL (accessing .md or .html files directly)

### Issue: Styles not loading
**Solution**: Make sure you're at the base URL, not a specific file.

## What to Check

1. **Is GitHub Pages enabled?** 
   - Settings → Pages → Source should be set

2. **Is the build succeeding?**
   - Actions tab → should see green checkmarks

3. **Are you using the right URL?**
   - Should end with just `/Tutorials/` not `/Tutorials/index.html`

## If Nothing Works

Try this reset:

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials

# Create a simple .nojekyll file to test if Pages is working
touch .nojekyll
git add .nojekyll
git commit -m "Test GitHub Pages"
git push origin main

# Wait 2 minutes then visit:
# https://omaroua.github.io/Tutorials/
```

If you see ANYTHING (even if broken), Pages is working. Then:

```bash
# Remove .nojekyll to let Jekyll run
rm .nojekyll
git add .nojekyll
git commit -m "Re-enable Jekyll"
git push origin main
```

## Screenshot Your Settings

Take screenshots of:
1. Settings → Pages (source settings)
2. Actions tab (build status)
3. What you see in the browser (the "HTML" page)

This will help diagnose the issue!

## Alternative: Use GitHub Pages Source  as "GitHub Actions"

If the above doesn't work:

1. Settings → Pages
2. Change Source from "Deploy from branch" to **"GitHub Actions"**
3. GitHub will suggest a Jekyll workflow - click **Configure**
4. Commit the workflow file
5. Wait for it to build

This gives you more control and better error messages!

