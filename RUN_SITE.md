# How to Run Your Tutorial Website

## The Problem

You're seeing bare HTML because you're opening the files directly in your browser. This is a **Jekyll website** that needs to be served through Jekyll to show the beautiful styling!

## Quick Solution (Recommended)

### Option 1: Deploy to GitHub Pages (Easiest - No Local Setup Needed!)

1. **Push your code to GitHub:**
   ```bash
   cd /Users/marouaoukrid/Desktop/Github/Tutorials
   git add .
   git commit -m "Updated tutorial website styling"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub: https://github.com/OMaroua/Tutorials
   - Click **Settings** → **Pages**
   - Under "Source", select **main** branch
   - Click **Save**
   - Wait 2-3 minutes, then visit: https://OMaroua.github.io/Tutorials/

🎉 **GitHub Pages will automatically run Jekyll and your site will look perfect!**

---

### Option 2: Run Jekyll Locally (Requires Setup)

If you want to preview locally before pushing to GitHub:

#### Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

Click "Install" when the popup appears. This may take 10-15 minutes.

#### Step 2: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 3: Install Ruby via Homebrew

The system Ruby (2.6) is too old and causes permission issues. Install a newer version:

```bash
brew install ruby

# Add to your PATH (add this to ~/.zshrc)
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/opt/homebrew/lib/ruby/gems/3.3.0/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify the new Ruby is being used
which ruby
ruby -v  # Should show 3.x.x, not 2.6.x
```

#### Step 4: Install Bundler and Jekyll

```bash
gem install bundler jekyll
```

#### Step 5: Install Dependencies

```bash
cd /Users/marouaoukrid/Desktop/Github/Tutorials
bundle install
```

#### Step 6: Run the Site!

```bash
bundle exec jekyll serve
```

Open your browser to: http://localhost:4000/Tutorials/

---

## What You'll See

Once the site is running (either on GitHub Pages or locally), you'll see:

✨ **Beautiful Features:**
- Modern gradient header with sticky navigation
- Clean, readable typography optimized for learning
- Syntax-highlighted code blocks
- Math equations beautifully rendered with MathJax
- Floating table of contents (on large screens)
- Smooth scrolling
- Responsive design for mobile/tablet
- Professional styling throughout

---

## Troubleshooting

### "I don't want to install Ruby/Jekyll locally"

→ **Use Option 1 (GitHub Pages)!** It's much easier and the site will look the same.

### "GitHub Pages is showing a 404"

→ Make sure you:
1. Pushed your code to GitHub
2. Enabled Pages in Settings
3. Selected the main branch as source
4. Waited 2-3 minutes for GitHub to build the site

### "Ruby is still showing 2.6.x after installing via Homebrew"

→ Make sure you added the PATH exports to `~/.zshrc` and ran `source ~/.zshrc`

### "I see errors about 'commonmarker' when running bundle install"

→ Install Xcode Command Line Tools: `xcode-select --install`

---

## Why Not Just Open the HTML Files?

Jekyll does these important things:
1. Compiles `.scss` → `.css` (your styles)
2. Processes Markdown → HTML
3. Applies layouts and templates
4. Handles navigation and URLs correctly
5. Renders math with MathJax
6. Processes code syntax highlighting

Opening files directly (`file:///...`) bypasses all of this!

---

## Recommended Approach

**For most users:** Use GitHub Pages (Option 1). It's:
- ✅ Zero setup required
- ✅ Automatically updated when you push
- ✅ Free hosting
- ✅ Fast and reliable
- ✅ Accessible from anywhere

**Only use Option 2** if you need to preview changes before pushing to GitHub.

---

## Your Site is Already Beautiful!

The styling work is done! The site includes:
- 🎨 Modern color scheme with smooth gradients
- 📖 Optimized typography for reading math/code
- 🧮 Beautiful math equation rendering
- 💻 Dark-themed code blocks with syntax highlighting
- 📱 Fully responsive design
- ♿ Accessibility features

You just need to run it through Jekyll to see it! 🚀

---

**Questions?** Check QUICKSTART.md for more details!

