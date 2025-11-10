# Math Equation Rendering Fix

## Issue

Inline math equations (using `$...$`) may not render properly and show as raw LaTeX code.

## What I Fixed

1. **Changed Kramdown input** from `GFM` to `kramdown` - this allows proper math processing
2. **Updated MathJax configuration** - using tex-svg renderer with better options
3. **Added enhanced MathJax processing** - ensures equations are rendered after page load

## If Inline Math Still Doesn't Work

Kramdown (Jekyll's markdown processor) has specific requirements for math:

### Option 1: Use Double Dollar Signs (Recommended)

**For inline math:** Use `$$..$$` instead of `$...$`

```markdown
The equation $$\mathbf{x} \sim p_\theta(\mathbf{x})$$ shows...
```

### Option 2: Use LaTeX Delimiters

**For inline math:** Use `\(...\)`

```markdown
The equation \(\mathbf{x} \sim p_\theta(\mathbf{x})\) shows...
```

**For display math:** Use `\[...\]`

```markdown
\[
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
\]
```

### Option 3: Use Kramdown Math Blocks

**For display math:**

```markdown
$$
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
$$
```

## Quick Fix Script

If you want to batch-convert single `$` to `$$` in your tutorial files:

```bash
# Backup first!
cp 01-VAE-Tutorial/tutorial.md 01-VAE-Tutorial/tutorial.md.backup

# Convert inline math (be careful - this is a simple regex)
# You may need to do this manually for accuracy
sed -i '' 's/\$\([^$]*\)\$/$$\1$$/g' 01-VAE-Tutorial/tutorial.md
```

**WARNING:** This regex is simplistic and may cause issues. Manual editing is safer!

## Testing

After making changes:

1. **If using GitHub Pages:** 
   - Push changes
   - Wait 2-3 minutes
   - Refresh the page (Cmd+Shift+R for hard refresh)

2. **If running locally:**
   - Stop Jekyll (Ctrl+C)
   - Restart: `bundle exec jekyll serve`
   - Hard refresh browser (Cmd+Shift+R)

## Current Status

The configuration is now optimized for math rendering. If you still see raw LaTeX:

1. Check that you're using `$$...$$` (double dollars) for inline math
2. Make sure there's no space after opening `$$` or before closing `$$`
3. Display math (equations on their own line) should work with `$$` on separate lines

## Example Working Math

### Inline Math (use double dollars):
```markdown
The prior $$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$ is Gaussian.
```

### Display Math:
```markdown
$$
\log p_\theta(\mathbf{x}) = \mathcal{L}_{\theta,\phi}(\mathbf{x}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))
$$
```

## Why This Happens

Kramdown (Jekyll's Markdown processor) doesn't support single `$` delimiters when using GFM input mode. The fix:

- Changed to native kramdown input
- Kramdown recognizes `$$...$$` for both inline and display math
- MathJax is configured to process both

## Need Help?

If math still isn't rendering after trying these fixes:

1. Check browser console for MathJax errors (F12 → Console)
2. Verify MathJax is loading (should see "MathJax rendering complete" in console)
3. Make sure you're using `$$...$$` not `$...$`

---

**After applying these fixes, commit and push:**

```bash
git add .
git commit -m "Fixed math equation rendering"
git push origin main
```

Wait 2-3 minutes and hard-refresh your GitHub Pages site!

